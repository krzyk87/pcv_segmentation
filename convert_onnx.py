from LitUnet import LitUnet  # constructing NN models
import torch
# custom functions
from utils import encode_experiment_name, find_best_model

params = {
    'LEARNING_RATE': 0.000005,
    'network': 'FCN8s',  # UNet / LFUNet / AttUNet / DRUNet / ReLayNet / FCN8s
    'kernel': (3, 3),  # (3, 3) / (7, 3) / ...
    'features': 32,  # 16 / 32 / 64
    'augment': None,  # (["FlipH", "Translate", "Rotate", "CropRandom"]) # "CropFixed"
    'distance_map': None,  # ’BasicOrient’ / ’CumSum’ / ’2NetR' / ’2NetPR’
    'batch': 1,  # if data augmentation is used -> batch must 1, otherwise can be bigger
    'N_CLASSES': 4,  # to how many segments the OCT image is divided (1 more than searched borders)
    'anomalies': "anomalies50anona3.csv"
    # "anomalies50anona3.csv" # remove anomalous samples listed in .csv file from the dataset
}
params['experiment'] = 'k' + str(params['kernel'][0]) + str(params['kernel'][1])
experiment_name, version_name, save_model_path = encode_experiment_name(params)
best_model_path = find_best_model(save_model_path)
model = LitUnet.load_from_checkpoint(best_model_path, param=params, model_path=save_model_path)
model.eval()

print(" Wczytano model ")
x = torch.randn(1, 1, 640, 384)
torch.onnx.export(model,  # model being run
                  x,  # model input
                  "model.onnx",  # where to save the model
                  export_params=True,  # store the trained weights
                  opset_version=11
                  )
