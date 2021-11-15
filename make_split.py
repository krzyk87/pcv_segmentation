import glob
import os
from pathlib import Path
import random
import shutil
import json

from utils import parse_gt, remove_anomalous

DATASET_PATH = Path("dataset50")
GT_PATH = Path("mvri")
GT_DIR = Path("ground_truth")

random.seed(5376)

TRAIN_VAL_SPLIT = 0.2
VAL_TEST_SPLIT = 0.1

anomalies = None    # "anomalies50_5.csv"

if __name__ == '__main__':
    file_list = []
    print('Making a list of files...')
    for img in glob.glob(os.path.join(DATASET_PATH, "*", "*.tiff")):
        img_path = img.split(os.path.sep)[-2:]
        file_list.append(os.path.join(*img_path))

    if anomalies is not None:
        file_list = remove_anomalous(file_list, anomalies)

    split_dict = {
        "train": [],
        "val": [],
        "test": []
    }
    for img in file_list:
        print(img)

        random_value = random.random()
        if random_value > TRAIN_VAL_SPLIT:
            split_dict["train"].append(img)
        elif random_value > VAL_TEST_SPLIT:
            split_dict["val"].append(img)
        else:
            split_dict["test"].append(img)

    split_file_name = "split.json"
    with open(split_file_name, "w+") as split_outfile:
        split_outfile.write(json.dumps(split_dict))

    os.makedirs(GT_DIR, exist_ok=True)
    for gt_filepath in glob.glob(os.path.join(GT_PATH, "*.mvri")):
        print(gt_filepath)
        gt_file = parse_gt(gt_filepath)
        gt_out_filename = os.path.join(GT_DIR, gt_filepath.split(os.path.sep)[-1].replace("mvri", "json"))
        with open(gt_out_filename, "w+") as gt_outfile:
            print(gt_out_filename)
            gt_outfile.write(json.dumps(gt_file))
