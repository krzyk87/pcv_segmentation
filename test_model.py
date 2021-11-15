import os
from torch.utils.data import DataLoader

from train import my_transform_img, my_transform_gt
from PcvDataset import PcvDataset
from LitUnet import LitUnet
from utils import encode_experiment_name, find_best_model

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

anomalies = "anomalies50a3.csv"    # "anomalies50a3.csv"
params_test = {
    'LEARNING_RATE': 0.000005,
    'network': 'DRUNet4',
    'kernel': (3, 3),
    'features': 16,
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropFixed", "CropRandom"])
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,
    'N_CLASSES': 4,
    'line': False,
    'create_dm': None,
    'test_all': True,
    'pretrained': False,
    'graph_path': None,  # 'denoiseWST1_cutAdapt_trackILMandISOS_smooth',
    'anomalies': anomalies
}
params_test['experiment'] = 'k' + str(params_test['kernel'][0]) + str(params_test['kernel'][1]) + ''

experiment_name, version_name, save_model_path = encode_experiment_name(params_test)
if params_test['test_all']:
    tensorboard_logdir = 'tb_logs_testall'
else:
    tensorboard_logdir = 'tb_logs'

if __name__ == '__main__':
    testing_data = PcvDataset(img_dir="dataset50", gt_dir="ground_truth50", dataset_split=["split50.json", "test"],
                              param=params_test, trans_img=my_transform_img, trans_gt=my_transform_gt, istest=True,
                              anomaly=anomalies)

    testing_dataloader = DataLoader(testing_data, batch_size=1, num_workers=4)     # shuffle=True
    model_file_path = find_best_model(save_model_path)
    test_model = LitUnet.load_from_checkpoint(model_file_path, param=params_test, model_path=save_model_path)

    tensorboard_logger = TensorBoardLogger(tensorboard_logdir, name=experiment_name, version=version_name)
    trainer_test = pl.Trainer(gpus=1, logger=tensorboard_logger)

    trainer_test.test(test_model, testing_dataloader)
