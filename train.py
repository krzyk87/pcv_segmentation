import os

# custom classes:
from PcvDataset import PcvDataset   # loading images
from LitUnet import LitUnet         # constructing NN models

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import torchvision.transforms.functional as func

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# custom functions
from utils import encode_experiment_name, find_best_model

# learning parameters
params_train = {
    'LEARNING_RATE': 0.000005,
    'network': 'UNetOrg',   # UNetOrg / LFUNet / AttUNet / DRUNet / ReLayNet / FCN8s
    'kernel': (7, 3),       # (3, 3) / (7, 3) / ...
    'features': 32,         # 16 / 32 / 64
    'augment': ["FlipH", "Translate", "Rotate", "CropRandom"],        # (["FlipH", "Translate", "Rotate", "CropRandom"])  # "CropFixed"
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,             # if data augmentation is used -> batch must be 1, otherwise can be bigger
    'N_CLASSES': 10,        # to how many segments the OCT image is divided (1 more than searched borders)
    'anomalies': "anomalies50anona3.csv"    # "anomalies50anona3.csv"      # remove anomalous samples listed in .csv file from the dataset
}
params_train['experiment'] = 'k' + str(params_train['kernel'][0]) + str(params_train['kernel'][1]) + '_nc10'

experiment_name, version_name, save_model_path = encode_experiment_name(params_train)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

# image transformations for CAVRI dataset
my_transform_img = transforms.Compose([
    transforms.Normalize(0.0634, 0.0117),
])
my_transform_gt = transforms.Compose([])


# main function run to train the model
if __name__ == '__main__':
    # define list of images for training and validation (directories, train/val split, image transforms, anomalous data exclusion)
    training_data = PcvDataset(img_dir="dataset50anon", gt_dir="ground_truth50anon", param=params_train,
                               dataset_split=["split50win_anon.json", "train"], trans_img=my_transform_img,
                               trans_gt=my_transform_gt)
    validation_data = PcvDataset(img_dir="dataset50anon", gt_dir="ground_truth50anon", param=params_train,
                                 dataset_split=["split50win_anon.json", "val"], trans_img=my_transform_img,
                                 trans_gt=my_transform_gt, istest=True)

    # set params for pytorch dataloader
    training_dataloader = DataLoader(training_data, batch_size=params_train['batch'], shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=1)

    # define model from the LitUnet class
    segment_model = LitUnet(param=params_train, model_path=save_model_path)

    # define training parameters
        # monitor validation loss; if it doesn't improve after 5 epochs - stop training
    early_stopping = EarlyStopping('val_loss', patience=5)
        # save the best model; include the epoch number and validation loss value in the file name
    model_checkpoint = ModelCheckpoint(save_model_path, filename='{epoch}-{val_loss:.4f}')
        # log results of each epoch  with TensorBoard
    tensorboard_logger = TensorBoardLogger('tb_logs', name=experiment_name, version=version_name)   # log_graph=True
    trainer = pl.Trainer(gpus=1, callbacks=[early_stopping, model_checkpoint], logger=tensorboard_logger, min_epochs=30,
                         resume_from_checkpoint=find_best_model(save_model_path), max_epochs=100)

    # start model training
    trainer.fit(segment_model, training_dataloader, val_dataloader)
