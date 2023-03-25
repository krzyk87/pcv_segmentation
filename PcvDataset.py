import glob
import json
import os

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
from torchvision.transforms.functional import to_tensor, hflip, rotate, affine, to_pil_image
from torch.utils.data import Dataset
from skimage import io
from utils import visualize, get_scan_number, save_gt_tiff, calculate_distance_map, crop_image, one_hot_to_sum,\
    remove_anomalous, visualize_dm, calculate_weight_matrix
import random
# from tqdm import tqdm
from matplotlib import pyplot as plt

empty_trans = transforms.Compose([])


class PcvDataset(Dataset):
    def __init__(self, img_dir, gt_dir, param, dataset_split=None, trans_img=empty_trans, trans_gt=empty_trans, istest=False):
        self.img_dir = img_dir      # directory with OCT (.tiff) images
        self.gt_dir = gt_dir        # directory with corresponding ground truth segmentations (.json)
        self.n_classes = param['N_CLASSES']
        self.transform = trans_img
        self.target_transform = trans_gt

        if dataset_split is None:   # use all images in the folder
            self.image_list = list(self.list_dir(dirpath=self.img_dir))
        else:                       # use the images from the split list [train / val / test]
            self.image_list = self.list_files(dirpath=self.img_dir, split_file=dataset_split[0], split_set=dataset_split[1])

        self.anomaly = param['anomalies']
        if self.anomaly is not None:    # exclude anomalous samples from the dataset
            count_all = len(self.image_list)
            self.image_list = remove_anomalous(self.image_list, self.anomaly)
            count_new = len(self.image_list)
            print('Removing anomalies: ' + self.anomaly + f' ({count_all} --> {count_new})')

        self.num_images = len(self.image_list)
        self.network = param['network']
        self.distance_map = param['distance_map']
        self.kernel = param['kernel']
        self.augment_list = ["None" for x in range(self.num_images)]

        # if data augmentation is set for the TRAINING -> expand the image list with each augmentation method
        if param['augment'] is not None and not istest:
            for technique in param['augment']:
                for n in range(self.num_images):
                    self.image_list.append(self.image_list[n])
                    self.augment_list.append(technique)

    def __len__(self):
        return len(self.image_list)

    # read ground truth from file
    def read_gt(self, ann_file):
        with open(ann_file) as f:
            gt_dict = json.loads(f.read())
        return gt_dict

    # list all .tiff files in the directory
    def list_dir(self, dirpath):
        for f in sorted(glob.glob(os.path.join(dirpath, "*/*.tiff"))):
            yield f

    # get file list for a split subset
    def list_files(self, dirpath, split_file, split_set):
        with open(split_file) as f:
            split_dict = json.loads(f.read())
        set_list = split_dict[split_set]
        file_list = [os.path.join(dirpath, file_path) for file_path in set_list]
        return file_list

    # set pixels between lines as belonging to a given class
    def assign_region(self, nc, indices, gt_matrix):
        for c in range(nc):
            if c == 0:
                if indices[c] > -1:
                    gt_matrix[c, :indices[c]] = 255
            elif c < (nc-1):
                if (indices[c-1] > -1) & (indices[c] > -1):
                    gt_matrix[c, indices[c-1]:indices[c]] = 255
            else:
                if (indices[c-1] > -1):
                    gt_matrix[c, indices[c-1]:] = 255
        return gt_matrix

    # create ground truth image from data loaded from .json
    def make_retinaNc(self, gt, image, scan_number, nc):
        gt_image = np.zeros((nc, image.shape[0], image.shape[1]), dtype=np.uint8)
        ilm = gt.get("ILM")[scan_number]
        rpe = gt.get("OB_RPE")[scan_number]
        if nc == 4:
            pcv = gt.get("PCV")[scan_number]
        for x, yILM in enumerate(ilm):
            yRPE = rpe[x]
            if nc == 3:
                gt_image[:, :, x] = self.assign_region(nc, (yILM, yRPE), gt_image[:, :, x])
            elif nc == 4:
                yPCV = pcv[x]
                if yPCV > yILM:
                    yPCV = yILM
                gt_image[:, :, x] = self.assign_region(nc, (yPCV, yILM, yRPE), gt_image[:, :, x])
        return gt_image

    # get sample (image, ground truth, image_path)
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = io.imread(img_path, as_gray=True)
        scan_number = get_scan_number(img_path)
        gtpath = os.path.join(self.gt_dir, img_path.split(os.path.sep)[-2] + ".json")
        ground_truth = self.read_gt(ann_file=gtpath)
        gt_image = self.make_retinaNc(ground_truth, image, scan_number, self.n_classes)

        image = to_tensor(image[:, 1:])
        if self.n_classes > 1:
            gt_image = to_tensor(np.transpose(gt_image[:, :, 1:], (1, 2, 0)))
        else:
            gt_image = to_tensor(gt_image[:, 1:])

        # add distance map as second channel to the image (if distance map is used)
        if self.distance_map is not None:
            if '2Net' not in self.distance_map:     # use map based on image intensities
                distance_map = calculate_distance_map(image, method=self.distance_map)
            else:                                   # use map based on previous segmentations
                path_list = img_path.split(os.path.sep)
                path_list[0] += '_DM' + self.distance_map
                distance_map = io.imread(os.path.join(*path_list))
                distance_map = to_tensor(np.transpose(distance_map, (1, 2, 0)))
            # visualize_dm(image[0], distance_map[0])
            image = torch.cat((image, distance_map), dim=0)

        # apply transformations to image and ground truth (if provided)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt_image = self.target_transform(gt_image).to(torch.uint8)

        # perform data augmentation
        if self.augment_list[idx] == "FlipH":       # horizontal flip
            image = hflip(image)
            gt_image = hflip(gt_image)
        elif self.augment_list[idx] == "Rotate":    # random rotation +- 20 deg
            rotation_range = 20
            rotation_angle = random.random()*2*rotation_range - rotation_range
            image = rotate(image, rotation_angle, interpolation=func.InterpolationMode.BILINEAR)
            gt_image = rotate(gt_image, rotation_angle)
        elif self.augment_list[idx] == "CropFixed":     # crop image with fixed width and height
            image, gt_image = crop_image(image, gt_image, fixed=True)
        elif self.augment_list[idx] == "CropRandom":    # crop image with random width and height
            image, gt_image = crop_image(image, gt_image, fixed=False)
        elif self.augment_list[idx] == "Translate":     # translate image vertically by random height
            image_height = image.size()[1]
            translation_max = image_height/10
            translation_y = int(random.random()*2*translation_max - translation_max)
            image = affine(image, angle=0, scale=1.0, translate=[0, translation_y], shear=[0.0])
            gt_image = affine(gt_image, angle=0, scale=1.0, translate=[0, translation_y], shear=[0.0])

        # data visualization / saving to file ------------------
        # gt_sum = one_hot_to_sum(torch.unsqueeze(gt_image, dim=0))
        # weight = calculate_weight_matrix(torch.unsqueeze(gt_image, dim=0).cuda(), edge_weight=10, area_weights=(1, 5, 5, 1)).cpu()
        # visualize(image.numpy().squeeze(), weight.numpy().squeeze(), gt_sum.numpy().squeeze())
        # save_gt_tiff(self.gt_dir, img_path, gt_sum)

        sample = {"image": image, "mask": gt_image, "image_path": img_path}
        return sample


params_test = {
    'network': 'UNetOrg',
    'kernel': (3, 3),
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropRandom"])  # "CropFixed",
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'N_CLASSES': 4,
    'anomalies': None    # "anomalies50anona3.csv"
}

# function to test the loading and visualization of the data
if __name__ == '__main__':
    data = PcvDataset(img_dir="dataset50anon", gt_dir="ground_truth50anon", dataset_split=["split50win_anon.json", "val"],
                      param=params_test)

    file_path = os.path.sep.join(("dataset50anon", "VMA_0001S_20160715103809", "Skan_nr_72.tiff"))

    idx = data.image_list.index(file_path)
    img = data[idx].get("image")[0, :, :]
    # mask = data[idx].get("mask")
    figure = plt.figure(1)
    ax = plt.subplot()
    ax.imshow(img.numpy(), cmap='gray')
    plt.show()

    # # for it, s in enumerate(data):
    # #     print(f'Saved file {it}')
    #     # visualize(s.get("image")[0, :], s.get("mask")[0, :]+s.get("image")[0, :], s.get("mask")[0, :])
    #     # break
    pass
