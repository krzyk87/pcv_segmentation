import datetime
import os
from skimage import io
from pathlib import Path
import train
import torch
from utils import get_scan_number, logit_to_one_hot, get_prediction_lines, save_prediction_json, \
    calculate_distance_map, encode_experiment_name, find_best_model, get_prediction_lines_dict, save_pred_dict_json
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import time


def evaluate_model(scan_path, model_to_evaluate, distance_map=None):

    # get list of image in a 3D scan folder to evaluate
    file_list = list(os.listdir(scan_path))
    image_width = to_tensor(io.imread(os.path.join(scan_path, file_list[0]), as_gray=True)).size()[2] - 1
    file_number = len(file_list)

    eval_time = 0   # setup timer
    pred_lines = {"PCV": [], "ILM": [], "OB_RPE": []}   # list retina borders to predict
    lines = torch.zeros(file_number, len(pred_lines), image_width)
    with torch.no_grad():
        for file_name in tqdm(file_list):

            # load image
            image_path = os.path.join(scan_path, file_name)
            scan_number = get_scan_number(image_path)
            image = io.imread(image_path, as_gray=True)
            image = to_tensor(image[:, 1:])

            # add distance map as second channel to the image (if used)
            if distance_map is not None:
                image_map = calculate_distance_map(image, method=distance_map)  # TODO: model_to_eval=model_to_evaluate
                image = torch.cat((image, image_map), dim=0)

            # send data to GPU / CPU
            image = train.my_transform_img(image).to(device)

            for i in range(1):
                start = datetime.datetime.now()  # time.time()
                prediction = model_to_evaluate(torch.unsqueeze(image, dim=0))   # predict segmentation for the image
                stop = datetime.datetime.now()
                eval_time += ((stop - start).total_seconds() * 1000)       # time.time() - start

            # extract segmentation borders from the result
            th = 0.5
            pred = logit_to_one_hot(prediction, th)
            lines[scan_number] = get_prediction_lines(pred.int())
            # pred_lines = get_prediction_lines_dict(pred.int(), pred_lines)

        # save segmentation data for a whole 3D image to a .json file
        save_prediction_json(lines, scan_path)
        # save_pred_dict_json(pred_lines, scan_path)
        total_time = eval_time / 1000   # sum(eval_time)/len(eval_time)/1000
        print(f'Evaluation time: {total_time}')
    return total_time


# provide folder path with new 3D OCT directories (with .tiff files) to segment
data_path = Path("new_predictions")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup parameters of the trained model to use for prediction
params_new = {
    'LEARNING_RATE': 0.000005,
    'network': 'UNetOrg',
    'kernel': (7, 3),
    'features': 32,
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropFixed", "CropRandom"])
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,
    'N_CLASSES': 4,
    'anomalies': "anomalies50a3anon.csv"
}
params_new['experiment'] = 'k' + str(params_new['kernel'][0]) + str(params_new['kernel'][1])
experiment_name, version_name, save_model_path = encode_experiment_name(params_new)

if __name__ == '__main__':
    # get list of 3D OCT scans from the directory
    scan_list = list(os.listdir(data_path))

    # search for model trained with selected parameters
    model_path = find_best_model(save_model_path)
    model = train.LitUnet.load_from_checkpoint(model_path, param=params_new)
    model.to(device)
    model.eval()

    # start predictions
    time_3d = 0
    time_3dimage = 0
    scan_count = len(scan_list)
    for scan_name in scan_list:
        if os.path.isdir(os.path.join(data_path, scan_name)):
            print(f'Evaluating scan: {scan_name}')
            start_image = datetime.datetime.now()
            time_3d += evaluate_model(os.path.join(data_path, scan_name), model, distance_map=None)
            stop_image = datetime.datetime.now()
            time_3dimage += ((stop_image - start_image).total_seconds() * 1000)
    print(f'Scans: {scan_count}')
    print(f'Time 3d: {time_3d}, avg: {time_3d/scan_count}')
    print(f'Time 3d image: {time_3dimage/1000},  avg: {time_3dimage/scan_count/1000}')
