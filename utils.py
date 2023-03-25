import csv
import os
import io
from pathlib import Path
import json

import scipy.ndimage.morphology
# import torchvision.utils
from lxml import etree
from matplotlib import pyplot as plt
import re
from tifffile import imwrite
import numpy as np
import torch
# import torchvision.transforms.functional as func
from torchvision.transforms.functional import crop, to_tensor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow

# import math
# from skimage.measure import label, regionprops
from skimage.morphology import disk    # opening, skeletonize
from skimage import filters, feature
from scipy import ndimage, optimize
import pandas as pd
# from pymorph import mmorph
import kornia

import random

GT_FILE = Path("gt.json")
GT_PATH = Path("mvri")

INTERESTING_SURFACES = (
    "PCV", "ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "INL-OPL", "BMEIS", "IS/OSJ", "IB_RPE", "OB_RPE")


def encode_experiment_name(params):
    experiment_name = '_'.join((params['network'], params['experiment']))
    if params['anomalies'] is not None:
        str_list = params['anomalies'].split('.')
        substr = str_list[0].replace('anomalies50anon', '')
        experiment_name = '_'.join((experiment_name, substr))
    if params['features'] != 64:
        experiment_name = '_'.join((experiment_name, str(params['features'])))
    if params['distance_map'] is not None:
        experiment_name = '_'.join((experiment_name, 'DM' + params['distance_map']))
    if params['augment'] is not None:
        for technique in params['augment']:
            experiment_name = '_'.join((experiment_name, technique))

    version_name = ''
    if params['batch'] != 1:
        version_name = 'B' + str(params['batch']) + '_'
    version_name += "lr={:.6f}".format(params['LEARNING_RATE'])
    save_model_path = os.path.join("saved_model", experiment_name+"_"+version_name)
    return experiment_name, version_name, save_model_path


def find_best_model(model_path):
    files = os.listdir(model_path)
    model_names = [filename for filename in files if filename.endswith('.ckpt')]    # 'epoch=13-val_loss=0.2368.ckpt'
    model_losses = []
    for file_name in model_names:
        name_split = file_name.split("=")
        model_losses.append(float(name_split[2].replace('.ckpt', '')))
    if len(model_losses) > 0:
        file_id = model_losses.index(min(model_losses))
        model_file_path = os.path.join(model_path, model_names[file_id])
        print('Found model: ' + model_file_path)
    else:
        model_file_path = None
        print('No model found - learning from scratch...')
    return model_file_path


def find_gpu(searched_name='GeForce'):
    gpu_id = 0
    device_count = torch.cuda.device_count()
    if device_count > 1:
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            if searched_name in gpu_name:
                gpu_id = i
                break
    return gpu_id


def check_if_important(name):
    res = re.search(r'\((.*?)\)', name).group(1)
    return res


def parse_gt(filepath):
    tree = etree.parse(str(filepath))
    root = tree.getroot()
    out_dict = {}

    for surface in root:
        if not surface.tag == "surface":
            continue

        for elem in surface:
            if elem.tag == "name":
                current_layer = check_if_important(elem.text)
                if current_layer not in INTERESTING_SURFACES:
                    continue
                for bscan in surface:
                    outarr = []
                    if not bscan.tag == "bscan":
                        continue
                    for z in bscan:
                        outarr.append(int(z.text))
                    if current_layer not in out_dict:
                        out_dict[current_layer] = []
                    out_dict[current_layer].append(outarr)

    return out_dict


def get_scan_number(filepath):
    filename = filepath.split(os.path.sep)[-1]
    fname = filename.split(".")[0]
    fnumber = int(fname.split("_")[-1])
    return fnumber - 1


def logit_to_one_hot(prediction, th):
    activation_fn = torch.nn.Softmax2d()
    pred = activation_fn(prediction)
    return pred > th


def one_hot_to_sum(matrix):
    n_classes = matrix.size()[1]
    result = torch.zeros_like(matrix[:, 0, :, :], dtype=torch.float32)
    for c in range(n_classes):
        result += (matrix[:, c, :, :] > 0) * (c+1)/n_classes * 255
    return result


# def sum_to_one_hot(matrix):
#     result = torch.zeros()
#     return result


def crop_image(image, gt_image, fixed=True):
    image_height = image.size()[1]
    image_width = image.size()[2]
    if fixed:
        output_height, output_width = image_height - int(image_height/10*2), image_width - int(image_width/10*2)
    else:
        min_height, min_width = image_height * 0.8, image_width * 0.8
        output_height = random.random() * (image_height - min_height) + min_height
        output_width = random.random() * (image_width - min_width) + min_width
    output_height = int(output_height / 16) * 16
    output_width = int(output_width / 16) * 16
    crop_x = random.random() * (image_width - output_width)
    crop_y = random.random() * (image_height - output_height)
    cropped_image = crop(image, int(crop_y), int(crop_x), output_height, output_width)
    cropped_gt = crop(gt_image, int(crop_y), int(crop_x), output_height, output_width)
    return cropped_image, cropped_gt


def calculate_weight_matrix(ground_truth, edge_weight, area_weights):
    n_classes = ground_truth.size()[1]
    ce_mat_weights_all = torch.zeros_like(ground_truth[:, 0, :, :])
    for c in range(n_classes):
        true = ground_truth[:, c, :, :].contiguous()
        true = true.to(torch.int8)
        batch_size = true.size()[0]
        empty_row = torch.zeros(batch_size, 1, true.size()[2]).cuda()
        edges_top = torch.sub(true[:, 1:, :], true[:, :-1, :]).clamp(0, 1)
        edges_top = torch.cat((empty_row, edges_top), dim=1)
        edges_bottom = torch.sub(true[:, :-1, :], true[:, 1:, :]).clamp(0, 1)
        edges_bottom = torch.cat((edges_bottom, empty_row), dim=1)
        edges = edges_top + edges_bottom

        true_no_edges = true.clone()
        true_no_edges[torch.gt(edges, 0)] = 0
        matrix = torch.add(torch.mul(torch.gt(edges, 0), edge_weight),
                           torch.mul(true_no_edges, area_weights[c]))
        ce_mat_weights_all.add_(matrix)

    return ce_mat_weights_all


# ------------ scores
def dice_coef(gt, pred):
    common_area = gt & pred
    number_elem_gt = torch.count_nonzero(gt)
    number_elem_pred = torch.count_nonzero(pred)
    number_elem_similar = torch.count_nonzero(common_area)
    return (2 * number_elem_similar) / (number_elem_gt + number_elem_pred)


def numeric_score(prediction, groundtruth):
    FP = (torch.sum((prediction == 1) & (groundtruth == 0))).float()
    FN = (torch.sum((prediction == 0) & (groundtruth == 1))).float()
    TP = (torch.sum((prediction == 1) & (groundtruth == 1))).float()
    TN = (torch.sum((prediction == 0) & (groundtruth == 0))).float()
    return FP, FN, TP, TN


def accuracy(FP, FN, TP, TN):
    N = FP + FN + TP + TN
    acc = (TP + TN) / N
    return acc


def precision(FP, FN, TP, TN):
    prec = torch.divide(TP, TP + FP)
    return prec * 100.0


def recall(FP, FN, TP, TN):
    rec = torch.divide(TP, TP + FN)
    return rec * 100.0


def f1score(FP, FN, TP, TN):
    f1 = torch.divide(2 * TP, 2 * TP + FP + FN)
    return f1


def calculate_metrics(prediction, groundtruth):
    th = 0.5
    pred = logit_to_one_hot(prediction, th)
    gt = (groundtruth > th).int()
    class_count = pred.size()[1]
    acc = torch.Tensor([])
    f1 = torch.Tensor([])
    for c in range(class_count):
        fp, fn, tp, tn = numeric_score(pred[:, c, :, :], gt[:, c, :, :])
        acc = torch.cat((acc, torch.Tensor([accuracy(fp, fn, tp, tn)])), dim=0)
        f1 = torch.cat((f1, torch.Tensor([f1score(fp, fn, tp, tn)])), dim=0)
    return acc.mean(), f1.mean()


def mean_abs_error(gt, pred):
    nc = pred.size()[1]
    img_width = gt.size()[3]
    mae = torch.Tensor([])

    prev_line_pred = None
    prev_line_gt = None
    for c in range(nc-1, 0, -1):
        pred_line = get_bottom_line(pred[:, c, :, :], prev_line_pred)
        gt_line = get_bottom_line(gt[:, c, :, :], prev_line_gt)
        prev_line_pred = pred_line
        prev_line_gt = gt_line

        ae_sum = 0.
        count_points = 0.
        for x in range(img_width):
            if (gt_line[x] != 0) & (pred_line[x] != 0):
                ae_sum += abs(gt_line[x] - pred_line[x])
                count_points += 1
        if count_points == 0:
            count_points = 1
        mae_line = ae_sum / count_points
        mae = torch.cat((torch.Tensor([mae_line]), mae), dim=0)
    return mae


def check_topology(pred):
    pred_sum = torch.unsqueeze(one_hot_to_sum(pred), 0)
    # diff = ((pred_sum[1:, :] - pred_sum[:-1, :]) < 0) & (pred_sum[1:, :] != 0)
    disc_kernel = torch.Tensor(disk(3)).cuda()
    pred_cor = kornia.morphology.opening(pred_sum, disc_kernel)
    pred_cor = kornia.morphology.closing(pred_cor, disc_kernel)
    pred_cor = torch.squeeze(pred_cor)
    diff_open = ((pred_cor[1:, :] - pred_cor[:-1, :]) < 0) & (pred_cor[1:, :] != 0)
    is_incorrect = 0
    if torch.sum(diff_open) > 0:
        is_incorrect = 1
        print(f'Incorrect topology! {torch.sum(diff_open)}')
        # pred_sum = torch.squeeze(pred_sum)
        # ax = plt.subplot(121)
        # ax.imshow(pred_sum.cpu())
        # ax = plt.subplot(122)
        # ax.imshow(pred_open.cpu())
        # plt.show()
    return is_incorrect


# --------------- post-processing
def get_top_line(input_matrix, upper_line=None):
    image_width = input_matrix.size()[2]
    top_line = torch.zeros(image_width)
    if upper_line is not None:
        for x in range(image_width):
            top_line[x] = upper_line[x] + torch.argmax(input_matrix[:, upper_line[x]:, x], dim=1)
    else:
        top_line = torch.argmax(input_matrix, dim=1)
        top_line = torch.squeeze(top_line)
    return top_line.long()


def get_bottom_line(input_matrix, lower_line=None):
    image_width = input_matrix.size()[2]
    image_height = input_matrix.size()[1]
    input_matrix = torch.flip(input_matrix, dims=[1])
    bottom_line = torch.zeros(image_width, device=input_matrix.device)
    if lower_line is not None:
        lower_line = image_height - lower_line
        for x in range(image_width):
            bottom_line[x] = lower_line[x] + torch.argmin(input_matrix[:, lower_line[x]:, x], dim=1)
    else:
        bottom_line = torch.argmin(input_matrix, dim=1)
        bottom_line = torch.squeeze(bottom_line)
    bottom_line = image_height - bottom_line
    return bottom_line.long()


def correct_prediction(pred):
    pred = pred.int()
    nc = pred.size()[1]
    image_width = pred.size()[3]
    new_pred = torch.zeros_like(pred)

    # topology_error = check_topology(pred)

    line0 = get_top_line(pred[:, 0, :, :]).long()
    line1 = get_top_line(pred[:, 1, :, :]).long()
    line2 = get_top_line(pred[:, 2, :, :], line1).long()
    for x in range(image_width):
        if line1[x] != 0:
            new_pred[:, 0, :line1[x], x] = 1
        else:
            new_pred[:, 0, :line2[x], x] = 1

    for c in range(1, nc-1):
        line_upper = get_top_line(pred[:, c, :, :], line0).long()
        line_lower = get_top_line(pred[:, c+1, :, :], line_upper).long()

        for x in range(image_width):
            if (line_upper[x] != 0) & (line_lower[x] != 0):
                new_pred[:, c, line_upper[x]:line_lower[x], x] = 1
        line0 = line_upper

    line_last = get_top_line(pred[:, nc-1, :, :], line0).long()
    for x in range(image_width):
        new_pred[:, nc-1, line_last[x]:, x] = 1

    return new_pred


# -------------- saving results
def save_loss(filename, epoch, train_loss_value):
    with open(filename, "a+", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writerow({"epoch": epoch, "train_loss": train_loss_value.item()})


def save_dice(filename, dice_scores):
    with open(filename, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["dice1", "dice2", "dice3", "dice4"])
        writer.writerow({"dice1": dice_scores[0].item(), "dice2": dice_scores[1].item(), "dice3": dice_scores[2].item(),
                         "dice4": dice_scores[3].item()})


def save_mae(filename, mae_values):
    with open(filename, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["maePCV", "maeILM", "maeRPE"])
        writer.writerow(
            {"maePCV": mae_values[0].item(), "maeILM": mae_values[1].item(), "maeRPE": mae_values[2].item()})


def save_acc(filename, acc_values):
    with open(filename, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["acc1", "acc2", "acc3", "acc4"])
        writer.writerow({"acc1": acc_values[0].item(), "acc2": acc_values[1].item(), "acc3": acc_values[2].item(),
                         "acc4": acc_values[3].item()})


def get_prediction_lines(prediction):
    n_classes = prediction.size()[1]
    lines = torch.zeros(n_classes-1, prediction.size()[3])
    line = None
    for c in range(n_classes-1, 0, -1):
        line = get_bottom_line(prediction[:, c, :, :], line)
        lines[c-1] = line
    return lines    # pred_pcv, pred_ilm, pred_rpe


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def get_prediction_lines_dict(prediction, out_dict):
    n_classes = prediction.size()[1]
    line = None
    for c in range(n_classes-1, 0, -1):
        line = get_bottom_line(prediction[:, c, :, :], line)
        out_dict[get_nth_key(out_dict, c-1)].append(line.tolist())
    return out_dict


def save_prediction_json(predictions, scan_path):
    print(f'Saving file: {scan_path}')
    n_classes = predictions.size()[1]
    out_dict = {"PCV": [], "ILM": [], "OB_RPE": []}
    for it in range(predictions.size()[0]):
        for c in range(n_classes):
            out_dict[get_nth_key(out_dict, c - 1)].append(predictions[it, c].tolist())
    pred_filename = scan_path + ".json"
    with open(pred_filename, "w+") as gt_outfile:
        gt_outfile.write(json.dumps(out_dict))


def save_pred_dict_json(pred_dict, scan_path):
    pred_filename = scan_path + ".json"
    with open(pred_filename, "w+") as gt_outfile:
        gt_outfile.write(json.dumps(pred_dict))


# -------------- visualize results
def visualize(img, prediction, ground_truth, corrected_prediction=None, scan_name="", save_to_file=False):
    figure = plt.figure(1)
    ax = plt.subplot(141)
    ax.imshow(img)
    ax.set_title("Image")
    ax = plt.subplot(142)
    ax.imshow(ground_truth, cmap='gray')
    ax.set_title("Ground Truth")
    ax = plt.subplot(143)
    img = ax.imshow(prediction)
    # plt.colorbar(img, ax=ax)
    ax.set_title("Prediction")
    ax = plt.subplot(144)
    if corrected_prediction is not None:
        ax.imshow(corrected_prediction)
        ax.set_title("Corrected prediction")
    else:
        # ax.imshow(abs(ground_truth - prediction) > 1)
        # ax.imshow(ground_truth ^ prediction)
        ax.set_title("Error")
    plt.suptitle(scan_name)

    if save_to_file:
        return figure
    else:
        plt.show()


def visualize_dm(img, dm):
    min_val = torch.min(dm)
    max_val = torch.max(dm)
    print(f'Min: {min_val}, max: {max_val}')

    # ax = plt.subplot(121)
    # ax.imshow(img)
    ax = plt.subplot()
    im = ax.imshow(dm)  # , vmin=-5, vmax=5
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(im)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig("dm2NetPRcolor.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tensorflow.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tensorflow.expand_dims(image, 0)
    return image


def plot_dice_boxplot(dice_table):
    dice_table = torch.transpose(dice_table, 0, 1)
    tick_pos = []
    tick_val = []
    if dice_table.size()[1] == 4:
        tick_pos = [1, 2, 3, 4]
        tick_val = ['Over PCV', 'PCV-ILM', 'ILM-RPE', 'Below RPE']
    elif dice_table.size()[1] == 9:
        tick_pos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tick_val = ['Over ILM', 'ILM-RNFL/GCL', 'GCL+IPL', 'INL', 'OPL', 'ONL', 'IS', 'OS+RPE', 'Below RPE']
    elif dice_table.size()[1] == 10:
        tick_pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tick_val = ['Over PCV', 'PCV-ILM', 'ILM-RNFL/GCL', 'GCL+IPL', 'INL', 'OPL', 'ONL', 'IS', 'OS+RPE', 'Below RPE']
    else:
        tick_pos = [1, 2, 3]
        tick_val = ['Over ILM', 'ILM-RPE', 'Below RPE']

    plt.figure(1)
    ax = plt.subplot(121)
    ax.violinplot(np.array(dice_table), showmedians=True)
    plt.xticks(tick_pos, tick_val)
    plt.ylim(-0.05, 1.05)

    ax = plt.subplot(122)
    ax.boxplot(np.array(dice_table))
    plt.xticks(tick_pos, tick_val)
    plt.ylim(-0.05, 1.05)
    plt.show()


# save ground truth as grayscale image
def save_gt_tiff(gt_path, img_path, gt_image):
    scan_folder = os.path.join(gt_path, img_path.split(os.path.sep)[-2])
    if not os.path.exists(scan_folder):
        os.mkdir(scan_folder)
    filename = os.path.join(scan_folder, img_path.split(os.path.sep)[-1])
    imwrite(filename, gt_image.numpy())  # to do: change float32 to uint8 for space saving


# merge image, ground truth, prediction and error into one matrix for visualization
def image_grid(image, gt, pred):
    if image.size()[0] > 1:
        image = image[0]
        image = torch.unsqueeze(image, dim=0)
    grid = torch.zeros((4, 1, image.size()[1], image.size()[2]))
    gt_sum = one_hot_to_sum(torch.unsqueeze(gt, 0))
    no_class = gt_sum == 0
    pred_one_hot = logit_to_one_hot(torch.unsqueeze(pred, 0), 0.5)
    pred_sum = torch.squeeze(one_hot_to_sum(pred_one_hot), 0)
    error = (abs(gt_sum - pred_sum) > 1).float() * 255
    if torch.sum(no_class) > 0:
        for y in range(error.size()[1]):
            for x in range(error.size()[2]):
                if no_class[0, y, x]:
                    error[0, y, x] = 0
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image)) * 255
    grid[0] = image
    grid[1] = gt_sum
    grid[2] = pred_sum
    grid[3] = error
    return grid / 255.0


# -------------- image statistics
def calculate_retina_orientation(retina_image):
    image_width = retina_image.size()[1]
    cut10 = int(image_width / 10)
    retina_blur = ndimage.gaussian_filter(retina_image[:, cut10:-cut10], sigma=3)
    # fig, ax = plt.subplots()
    edge_map_h = filters.sobel_h(retina_blur)
    # ax.imshow(edge_map_h, cmap=plt.cm.gray)
    edge_map_v = filters.sobel_v(retina_blur)
    angle = np.arctan2(edge_map_v, edge_map_h)

    orientation = -angle.mean() * 2
    return orientation  # *360/(2*np.pi)


def find_mass_center(retina_image):
    retina_image = (retina_image - torch.min(retina_image)) / (torch.max(retina_image) - torch.min(retina_image))
    cy, cx = ndimage.measurements.center_of_mass(np.array(retina_image))
    # fig, ax = plt.subplots()
    # ax.imshow(retina_image, cmap=plt.cm.gray)
    # ax.plot(cx, cy, '+r')
    # plt.close()
    return torch.tensor([cy, cx])


def calculate_distance_map(image, method='BasicOrient', prediction=None):
    image_height = image.size()[1]
    image_width = image.size()[2]
    mass_center = torch.zeros(1, 2)
    distance_map = torch.zeros_like(image)
    # fig = plt.figure(1)
    # ax = plt.subplot(131)
    # ax.imshow(torch.squeeze(image), cmap=plt.cm.gray)

    if method == 'BasicOrient':
        orientation = - calculate_retina_orientation(torch.squeeze(image))
        mass_center[0] = find_mass_center(torch.squeeze(image))
        binary_mask = torch.ones_like(image)
        for x in range(image_width):
            y = int(orientation * (x - mass_center[0, 1]) + mass_center[0, 0])
            binary_mask[:, y, x] = 0

        distance_map = scipy.ndimage.morphology.distance_transform_edt(binary_mask)
        distance_map = to_tensor(np.transpose(distance_map, (1, 2, 0)))

        line_indexes = torch.squeeze(torch.argmin(binary_mask, dim=1))
        for x in range(image_width):
            distance_map[:, :line_indexes[x], x] = -distance_map[:, :line_indexes[x], x]
        top_value = torch.min(distance_map[:, 0, :])
        distance_map -= top_value
        distance_map /= torch.max(distance_map)
        # ax = plt.subplot(142)
        # ax.imshow(torch.squeeze(distance_map), cmap=plt.cm.gray)
    elif method == 'CumSum':
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        cs = np.cumsum(image, axis=1)
        # ax = plt.subplot(142)
        # ax.imshow(torch.squeeze(cs), cmap=plt.cm.gray)

        smoothed = ndimage.gaussian_filter(cs, sigma=20)
        smoothed = to_tensor(np.transpose(smoothed, (1, 2, 0)))
        # ax = plt.subplot(142)
        # ax.imshow(torch.squeeze(smoothed), cmap=plt.cm.gray)

        scaled = (smoothed - 5) / 25.0
        # ax = plt.subplot(143)
        # ax.imshow(torch.squeeze(scaled), cmap=plt.cm.gray)

        # clamped = torch.clamp(scaled, 0, 1)
        # ax = plt.subplot(144)
        # ax.imshow(torch.squeeze(clamped), cmap=plt.cm.gray)
        # plt.show()
        distance_map = scaled
    elif method == '2NetR':
        pred_rpe = get_bottom_line(prediction[:, 3, :, :])
        pred_ilm = get_bottom_line(prediction[:, 2, :, :], pred_rpe)
        retina_distance = pred_rpe - pred_ilm
        for x in range(image_width):
            distance_map[:, :, x] = (torch.arange(end=image_height, device=prediction.device) - pred_ilm[x]) / retina_distance[x]
        # ax = plt.subplot(132)
        # ax.imshow(torch.squeeze(distance_map), cmap=plt.cm.gray)
        # clamped = torch.clamp(distance_map, 0, 1)
        # ax = plt.subplot(133)
        # ax.imshow(torch.squeeze(clamped), cmap=plt.cm.gray)
        # plt.show()
    elif method == '2NetPR':
        rpe = get_bottom_line(prediction[:, 3, :, :])
        ilm = get_bottom_line(prediction[:, 2, :, :], rpe)
        pcv = get_bottom_line(prediction[:, 1, :, :], ilm)
        pcv_rpe = rpe - pcv
        pcv_ilm = ilm - pcv
        ilm_rpe = rpe - ilm
        for x in range(image_width):
            distance_map[:, :pcv[x], x] = (torch.arange(pcv[x], device=prediction.device) - pcv[x]) / pcv_rpe[x]
            distance_map[:, pcv[x]:ilm[x], x] = 1/2 * (torch.arange(pcv[x], ilm[x], device=prediction.device) - pcv[x]) / pcv_ilm[x]
            distance_map[:, ilm[x]:rpe[x], x] = 1/2 * ((torch.arange(ilm[x], rpe[x], device=prediction.device) - ilm[x]) / ilm_rpe[x] + 1)
            distance_map[:, rpe[x]:, x] = (torch.arange(rpe[x], image_height, device=prediction.device) - pcv[x]) / pcv_rpe[x]
        # ax = plt.subplot(132)
        # ax.imshow(torch.squeeze(distance_map), cmap=plt.cm.gray)
        # clamped = torch.clamp(distance_map, 0, 1)
        # ax = plt.subplot(133)
        # ax.imshow(torch.squeeze(clamped), cmap=plt.cm.gray)
        # plt.show()

    return distance_map.to(torch.float32)


def remove_anomalous(image_list, anomaly_file):
    table = pd.read_csv(anomaly_file)
    path_list = image_list[0].split(os.path.sep)
    # folder_path = os.path.join(path_list[0], path_list[1])
    folder_path = path_list[0]
    for file_name in table['im2d_names']:
        scan = file_name.split('-')
        scan_name = scan[0]
        cross_section = scan[1]
        str1 = scan_name.split('.')
        part1 = '-'.join(str1[0:3])
        part2 = '.'.join(str1[3:-1])
        scan_name = '.'.join((part1, part2))
        image_path = os.path.join(folder_path, scan_name, 'Skan_nr_' + str(cross_section) + '.tiff')
        if image_path in image_list:
            image_list.remove(image_path)
        # indices = [i for i, s in enumerate(image_list) if image_path in s]
        # image_list.pop(indices)
    return image_list


# a function to test individual methods from this file
if __name__ == '__main__':
    dice_table = torch.zeros((4, 390))
    with open('saved_model/dice.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        it = 0
        for row in reader:
            next_row = torch.Tensor([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            dice_table[:, it] = next_row
            it += 1
    plot_dice_boxplot(dice_table)
    pass
