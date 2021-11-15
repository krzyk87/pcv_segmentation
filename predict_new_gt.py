import os
import json
from skimage import io
from pathlib import Path
import train
from PcvDataset import PcvDataset
import torch
from utils import get_scan_number, logit_to_one_hot, get_prediction_lines, save_prediction_json, \
    calculate_distance_map, encode_experiment_name, find_best_model
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import numpy as np


def evaluate_model(scan_path, model_to_evaluate, distance_map=None):
    file_list = list(os.listdir(scan_path))
    image_width = to_tensor(io.imread(os.path.join(scan_path, file_list[0]), as_gray=True)).size()[2] - 1
    file_number = len(file_list)
    pcv = torch.zeros(file_number, image_width)
    ilm = torch.zeros(file_number, image_width)
    rpe = torch.zeros(file_number, image_width)
    with torch.no_grad():
        for file_name in tqdm(file_list):
            image_path = os.path.join(scan_path, file_name)
            scan_number = get_scan_number(image_path)
            image = io.imread(image_path, as_gray=True)
            image = to_tensor(image[:, 1:])
            if distance_map is not None:
                if '2Net' not in distance_map:
                    distance_map = calculate_distance_map(image, method=distance_map)
                else:
                    path_list = image_path.split(os.path.sep)
                    path_list[0] += '_DM' + distance_map
                    distance_map = io.imread(os.path.join(*path_list))
                    distance_map = to_tensor(np.transpose(distance_map, (1, 2, 0)))
                image = torch.cat((image, distance_map), dim=0)
            image = train.my_transform_img(image).to(device)

            prediction = model_to_evaluate(torch.unsqueeze(image, dim=0))

            th = 0.5
            pred_one_hot = logit_to_one_hot(prediction, th).int()
            line_pcv, line_ilm, line_rpe = get_prediction_lines(pred_one_hot)
            pcv[scan_number] = line_pcv
            ilm[scan_number] = line_ilm
            rpe[scan_number] = line_rpe

        save_prediction_json([pcv, ilm, rpe], scan_path)


data_path = Path("dataset50")     # "new_predictions"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_new = {
    'LEARNING_RATE': 0.000005,
    'network': 'UNetOrg',
    'kernel': (9, 3),
    'features': 32,
    'augment': (["FlipH", "Translate", "Rotate", "CropRandom"]),        # (["FlipH", "Translate", "Rotate", "CropFixed", "CropRandom"])
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,
    'N_CLASSES': 4,
    'pretrained': False,
    'line': False,
    'anomalies': None
}
params_new['experiment'] = 'k' + str(params_new['kernel'][0]) + ''
experiment_name, version_name, save_model_path = encode_experiment_name(params_new)

if __name__ == '__main__':
    scan_list = list(os.listdir(data_path))

    model_path = find_best_model(save_model_path)
    model = train.LitUnet.load_from_checkpoint(model_path, param=params_new)
    model.to(device)
    model.eval()

    for scan_name in scan_list:
        if os.path.isdir(os.path.join(data_path, scan_name)):
            print(f'Evaluating scan: {scan_name}')
            evaluate_model(os.path.join(data_path, scan_name), model, distance_map=params_new['distance_map'])
