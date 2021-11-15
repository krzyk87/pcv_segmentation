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
from utils import visualize, get_scan_number, save_gt_tiff, calculate_distance_map, crop_image,\
    calculate_retina_orientation, plot_circ_distribution, one_hot_to_sum, find_mass_center, plot_mass_distribution,\
    remove_anomalous, save_orientations, save_center_mass, read_orientations, read_mass_centers, visualize_dm, \
    summary_box_plot
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

empty_trans = transforms.Compose([])


class PcvDataset(Dataset):
    def __init__(self, img_dir, gt_dir, param, dataset_split=None, trans_img=empty_trans, trans_gt=empty_trans, istest=False,
                 layers=("PCV", "ILM", "RNFL-GCL", "IPL-INL", "INL-OPL", "OPL-HFL", "BMEIS", "IS/OSJ", "IB_RPE", "OB_RPE"),
                 anomaly=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.layers = layers
        self.n_classes = param['N_CLASSES']
        self.transform = trans_img
        self.target_transform = trans_gt
        if dataset_split is None:
            self.image_list = list(self.list_dir(dirpath=self.img_dir))
        else:
            self.image_list = self.list_files(dirpath=self.img_dir, split_file=dataset_split[0], split_set=dataset_split[1])
        if anomaly is not None:
            count_all = len(self.image_list)
            self.image_list = remove_anomalous(self.image_list, anomaly)
            count_new = len(self.image_list)
            print('Removing anomalies: ' + anomaly + f' ({count_all} --> {count_new})')
        self.num_images = len(self.image_list)
        if param['network'] == 'Graph':
            self.read_graph = param['graph_path']
        else:
            self.read_graph = None
        self.distance_map = param['distance_map']
        self.network = param['network']
        self.kernel = param['kernel']
        self.segment_line = param['line']
        self.augment_list = ["None" for x in range(self.num_images)]
        if param['augment'] is not None and not istest:
            for technique in param['augment']:
                for n in range(self.num_images):
                    self.image_list.append(self.image_list[n])
                    self.augment_list.append(technique)

    def __len__(self):
        return len(self.image_list)

    def read_gt(self, ann_file):
        with open(ann_file) as f:
            gt_dict = json.loads(f.read())
        return gt_dict

    def list_dir(self, dirpath):
        for f in sorted(glob.glob(os.path.join(dirpath, "*/*.tiff"))):
            yield f

    def list_files(self, dirpath, split_file, split_set):
        with open(split_file) as f:
            split_dict = json.loads(f.read())
        set_list = split_dict[split_set]
        file_list = [os.path.join(dirpath, file_path) for file_path in set_list]
        return file_list

    def select_layers(self, nc):
        layers_list = []
        if (nc == 4) or (nc == 10):
            layers_list.append("PCV")
        layers_list.append("ILM")
        if nc > 8:
            layers_list.append("RNFL-GCL")
            layers_list.append("IPL-INL")
            layers_list.append("INL-OPL")
            layers_list.append("OPL-HFL")
            layers_list.append("BMEIS")
            layers_list.append("IS/OSJ")
        layers_list.append("OB_RPE")
        return layers_list

    def make_line(self, gt, image, scan_number, layer):
        gt_image = np.zeros_like(image)
        for x, y in enumerate(gt.get(layer)[scan_number]):  # line
            if y <= -1:
                continue
            gt_image[y, x] = 255
        return gt_image

    def assign_line(self, image, layer_line):
        gt_image = np.zeros_like(image)
        for x, y in enumerate(layer_line):  # line
            if y <= -1:
                continue
            gt_image[y, x] = 255
        return gt_image

    def make_lineNc(self, gt, image, scan_number, nc, layers):
        gt_image = np.zeros((nc, image.shape[0], image.shape[1]), dtype=np.uint8)
        for i, layer in enumerate(layers):
            layer_line = gt.get(layer)[scan_number]
            gt_image[i+1, :, :] = self.assign_line(image, layer_line)
        sum_nc = np.sum(gt_image, 0)
        gt_background = np.zeros_like(image)
        gt_background[sum_nc == 0] = 255
        gt_image[0, :, :] = gt_background
        return gt_image

    def make_retina(self, gt, image, scan_number):
        gt_image = np.zeros_like(image)
        ilm = gt.get("ILM")[scan_number]
        rpe = gt.get("OB_RPE")[scan_number]
        for x, y in enumerate(ilm):
            y2 = rpe[x]
            if y <= -1:
                continue
            if y2 <= -1:
                continue
            gt_image[y:y2, x] = 255
        return gt_image

    @staticmethod
    def assign_region(nc, indices, gt_matrix):
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

    def make_retinaNc(self, gt, image, scan_number, nc, layers):
        gt_image = np.zeros((nc, image.shape[0], image.shape[1]), dtype=np.uint8)
        ilm = gt.get("ILM")[scan_number]
        rpe = gt.get("OB_RPE")[scan_number]
        if (nc == 4) or (nc == 10):
            pcv = gt.get("PCV")[scan_number]
        if nc > 8:
            rnfl_gcl = gt.get("RNFL-GCL")[scan_number]
            ipl_inl = gt.get("IPL-INL")[scan_number]
            inl_opl = gt.get("INL-OPL")[scan_number]
            opl_onl = gt.get("OPL-HFL")[scan_number]
            bmeis = gt.get("BMEIS")[scan_number]
            is_os = gt.get("IS/OSJ")[scan_number]
            # irpe = gt.get("IB_RPE")[scan_number]
        for x, yILM in enumerate(ilm):
            yRPE = rpe[x]
            if (nc == 4) or (nc == 10):
                yPCV = pcv[x]
                if yPCV > yILM:
                    yPCV = yILM
            if nc >= 9:
                yNFLGCL = rnfl_gcl[x]
                yIPLINL = ipl_inl[x]
                yINLOPL = inl_opl[x]
                yOPLONL = opl_onl[x]
                yBMEIS = bmeis[x]
                yISOS = is_os[x]
            if nc == 3:
                gt_image[:, :, x] = self.assign_region(nc, (yILM, yRPE), gt_image[:, :, x])
            elif nc == 4:
                # if self.read_graph is not None:
                    # yILM = yILM + 1
                    # yPCV = yPCV + 1
                    # yRPE = yRPE + 1
                gt_image[:, :, x] = self.assign_region(nc, (yPCV, yILM, yRPE), gt_image[:, :, x])
            elif nc == 9:
                gt_image[:, :, x] = self.assign_region(nc, (yILM, yNFLGCL, yIPLINL, yINLOPL, yOPLONL, yBMEIS, yISOS, yRPE), gt_image[:, :, x])
            elif nc == 10:
                gt_image[:, :, x] = self.assign_region(nc, (yPCV, yILM, yNFLGCL, yIPLINL, yINLOPL, yOPLONL, yBMEIS, yISOS, yRPE), gt_image[:, :, x])
        return gt_image

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = io.imread(img_path, as_gray=True)
        scan_number = get_scan_number(img_path)
        gtpath = os.path.join(self.gt_dir, img_path.split(os.path.sep)[-2] + ".json")
        ground_truth = self.read_gt(ann_file=gtpath)
        layers_to_segment = self.select_layers(self.n_classes)
        if self.segment_line:
            gt_image = self.make_lineNc(ground_truth, image, scan_number, self.n_classes, layers_to_segment)
        else:
            gt_image = self.make_retinaNc(ground_truth, image, scan_number, self.n_classes, layers_to_segment)
        if self.read_graph is not None:
            graph_path = os.path.sep.join(("graph50", self.read_graph, img_path.split(os.path.sep)[-2] + ".json"))
            graph = self.read_gt(ann_file=graph_path)
            graph_pred = self.make_retinaNc(graph, image, scan_number, self.n_classes)
            graph_pred = to_tensor(np.transpose(graph_pred[:, :, 1:], (1, 2, 0)))
        else:
            graph_pred = "None"

        image = to_tensor(image[:, 1:])
        if self.n_classes > 1:
            gt_image = to_tensor(np.transpose(gt_image[:, :, 1:], (1, 2, 0)))
        else:
            gt_image = to_tensor(gt_image[:, 1:])

        if self.distance_map is not None:
            if '2Net' not in self.distance_map:
                distance_map = calculate_distance_map(image, method=self.distance_map)
            else:
                path_list = img_path.split(os.path.sep)
                path_list[0] += '_DM' + self.distance_map
                distance_map = io.imread(os.path.join(*path_list))
                distance_map = to_tensor(np.transpose(distance_map, (1, 2, 0)))
            # visualize_dm(image[0], distance_map[0])
            image = torch.cat((image, distance_map), dim=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt_image = self.target_transform(gt_image).to(torch.uint8)

        if self.augment_list[idx] == "FlipH":
            image = hflip(image)
            gt_image = hflip(gt_image)
        elif self.augment_list[idx] == "Rotate":
            rotation_range = 20
            rotation_angle = random.random()*2*rotation_range - rotation_range
            image = rotate(image, rotation_angle, interpolation=func.InterpolationMode.BILINEAR)
            gt_image = rotate(gt_image, rotation_angle)
        elif self.augment_list[idx] == "CropFixed":
            image, gt_image = crop_image(image, gt_image, fixed=True)
        elif self.augment_list[idx] == "CropRandom":
            image, gt_image = crop_image(image, gt_image, fixed=False)
        elif self.augment_list[idx] == "Translate":
            image_height = image.size()[1]
            translation_max = image_height/10
            translation_y = int(random.random()*2*translation_max - translation_max)
            image = affine(image, angle=0, scale=1.0, translate=[0, translation_y], shear=[0.0])
            gt_image = affine(gt_image, angle=0, scale=1.0, translate=[0, translation_y], shear=[0.0])

        # gt_sum = one_hot_to_sum(torch.unsqueeze(gt_image, dim=0))
        # graph_sum = one_hot_to_sum(torch.unsqueeze(graph_pred, dim=0))
        # visualize(image.numpy().squeeze(), graph_sum.numpy().squeeze(), gt_sum.numpy().squeeze())
        # save_gt_tiff(self.gt_dir, img_path, gt_sum)

        sample = {"image": image, "mask": gt_image, "image_path": img_path, "graph": graph_pred}
        return sample


params_test = {
    'network': 'UNetOrg',
    'kernel': (3, 3),
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropRandom"])  # "CropFixed",
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'graph_path': None,     # 'denoiseWST1_cutAdapt_trackILMandISOS',
    'line': True,
    'N_CLASSES': 4
}
anomalies = "anomalies50a3.csv"    # "anomalies50a5.csv"

if __name__ == '__main__':
    data = PcvDataset(img_dir="dataset50", gt_dir="ground_truth50", dataset_split=["split50.json", "train"],
                      param=params_test, anomaly=anomalies)
    pass
