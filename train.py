import os

from PcvDataset import PcvDataset
from LitUnet import LitUnet
# tensorboard --logdir=tb_logs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as func

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import encode_experiment_name, find_best_model, find_gpu

anomalies = "anomalies50a3.csv"    # "anomalies50a3.csv"
params_train = {
    'LEARNING_RATE': 0.000005,
    'network': 'DRUNet4',
    'kernel': (3, 3),
    'features': 16,
    'augment': None,        # (["FlipH", "Translate", "Rotate", "CropRandom"])  # "CropFixed"
    'distance_map': None,   # 'BasicOrient' / 'CumSum' / '2NetR' / '2NetPR'
    'batch': 1,
    'N_CLASSES': 4,
    'pretrained': False,
    'line': False,
    'anomalies': anomalies
}
params_train['experiment'] = 'k' + str(params_train['kernel'][0]) + str(params_train['kernel'][1]) + ''

experiment_name, version_name, save_model_path = encode_experiment_name(params_train)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

my_transform_img = transforms.Compose([
    transforms.Normalize(0.0634, 0.0117),
])
my_transform_gt = transforms.Compose([
])


if __name__ == '__main__':
    training_data = PcvDataset(img_dir="dataset50", gt_dir="ground_truth50", param=params_train,
                               dataset_split=["split50.json", "train"], trans_img=my_transform_img,
                               trans_gt=my_transform_gt, anomaly=anomalies)
    validation_data = PcvDataset(img_dir="dataset50", gt_dir="ground_truth50", param=params_train,
                                 dataset_split=["split50.json", "val"], trans_img=my_transform_img,
                                 trans_gt=my_transform_gt, anomaly=anomalies, istest=True)

    training_dataloader = DataLoader(training_data, batch_size=params_train['batch'], num_workers=4, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=1, num_workers=4)

    segment_model = LitUnet(param=params_train, model_path=save_model_path)

    early_stopping = EarlyStopping('val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(save_model_path, filename='{epoch}-{val_loss:.4f}')
    tensorboard_logger = TensorBoardLogger('tb_logs', name=experiment_name, version=version_name)   # log_graph=True
    gpu_id = find_gpu()
    trainer = pl.Trainer(gpus=[gpu_id], callbacks=[early_stopping, model_checkpoint], logger=tensorboard_logger, min_epochs=50,
                         resume_from_checkpoint=find_best_model(save_model_path))

    trainer.fit(segment_model, training_dataloader, val_dataloader)
