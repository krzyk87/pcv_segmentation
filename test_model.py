from torch.utils.data import DataLoader

from train import my_transform_img, my_transform_gt
from PcvDataset import PcvDataset
from LitUnet import LitUnet
from utils import encode_experiment_name, find_best_model

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# parameters of the experiment
params_test = {
    'LEARNING_RATE': 0.000005,
    'network': 'UNetOrg',   # UNetOrg / LFUNet / AttUNet / DRUNet / ReLayNet / FCN8s
    'kernel': (3, 3),       # (3, 3) / (7, 3)
    'features': 32,         # 16 / 32 / 64
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropFixed", "CropRandom"])
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,             # if data augmentation is used -> batch must be 1, otherwise -> the same as the training
    'N_CLASSES': 4,         # to how many segments the OCT image is divided (1 more than searched borders)
    'create_dm': None,      # create distance maps (code as in 'distance_map') for all images or test the model (None)
    'test_all': True,       # save all tested images to tensorboard or only 1/10 of tested images
    'anomalies': None  # "anomalies50anona3.csv"
}
params_test['experiment'] = 'k' + str(params_test['kernel'][0]) + str(params_test['kernel'][1])
experiment_name, version_name, save_model_path = encode_experiment_name(params_test)

if params_test['test_all']:     # set tensorboard directory
    tensorboard_logdir = 'tb_logs_testall'
else:
    tensorboard_logdir = 'tb_logs'


# main function run to test the trained model
if __name__ == '__main__':
    # define list of images for testing (directories, test subset (split), image transforms, anomalous data exclusion)
    testing_data = PcvDataset(img_dir="dataset50anon", gt_dir="ground_truth50anon", dataset_split=["split50win_anon.json", "test"],
                              param=params_test, trans_img=my_transform_img, trans_gt=my_transform_gt, istest=True)

    # set params for pytorch dataloader
    testing_dataloader = DataLoader(testing_data, batch_size=1)     # shuffle=True

    # find the best trained model with chosen parameters in the 'saved_model/' directory
    model_file_path = find_best_model(save_model_path)
    test_model = LitUnet.load_from_checkpoint(model_file_path, param=params_test, model_path=save_model_path)

    # log the test results with TensorBoard
    tensorboard_logger = TensorBoardLogger(tensorboard_logdir, name=experiment_name, version=version_name)
    trainer_test = pl.Trainer(gpus=1, logger=tensorboard_logger)

    # start the test of the model
    trainer_test.test(test_model, dataloaders=testing_dataloader)
