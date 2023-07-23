import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
from skimage import io
import pandas
# from matplotlib import pyplot as plt

from unet_model import UNetOrg, LFUNet, ReLayNet, FCN8s, AttUNet, DRUNet
from losses import DiceLoss, DiceCELoss, WeightedCrossEntropyLoss, BCEWithLogitsLoss
from utils import save_loss, calculate_metrics, image_grid, logit_to_one_hot, numeric_score, accuracy, dice_coef,\
    correct_prediction, mean_abs_error, save_dice, save_mae, save_acc, plot_dice_boxplot, calculate_distance_map, \
    one_hot_to_sum, check_topology

# weights vectors for unbalanced classes (computed with various methods)
weights_0 = torch.tensor([1.0, 1.0, 1.0, 1.0])
weights_1 = torch.tensor([0.09, 0.41, 0.36, 0.14])
weights_1a = torch.tensor([0.59, 0.91, 0.86, 0.64])
weights_2 = torch.tensor([1.5, 11.1, 6.0, 1.7])
weights_3 = torch.tensor([0.2, 1.0, 0.6, 0.23])
weights_4 = torch.tensor([1.0, 2.6, 2.0, 1.1])
weights_5 = torch.tensor([1.0, 4.9, 2.8, 1.1])
weights_5a = torch.tensor([0.1, 0.5, 0.29, 0.11])
weights_5a9c = torch.tensor([0.0044, 0.0927, 0.0765, 0.1966, 0.1505, 0.0619, 0.3292, 0.0825, 0.0058])
weights_5a10c = torch.tensor([0.0033, 0.2524, 0.0827, 0.0665, 0.1466, 0.1005, 0.0419, 0.2292, 0.0725, 0.0044])


# an overloaded custom Pytorch Lightning model
class LitUnet(pl.LightningModule):
    def __init__(self, param=None, model_path='saved_model'):
        super().__init__()
        self.model_path = model_path
        self.n_classes = param['N_CLASSES']

        image_channels = 1
        if 'create_dm' in param.keys():     # 'Create Distance Map' and save to file instead of testing the model
            self.create_dm = param["create_dm"]
        else:
            self.create_dm = None
        if (self.create_dm is None) and (param['distance_map'] is not None):
            image_channels = 2
        param['in_channels'] = image_channels

        # setup model architecture
        self.network = param['network']
        if param['network'] == 'LFUNet':
            self.model = LFUNet(params=param)
        elif param['network'] == 'ReLayNet':
            self.model = ReLayNet(params=param)
        elif param['network'] == 'FCN8s':
            self.model = FCN8s(params=param)
        elif param['network'] == 'AttUNet':
            self.model = AttUNet(params=param)
        elif param['network'] == 'DRUNet':
            self.model = DRUNet(params=param)
        else:
            self.model = UNetOrg(params=param)

        print('------------------------------------')
        print(f'Running {param["network"]}: features: {param["features"]}, kernel: {param["kernel"]}')
        print(f'Classes: {param["N_CLASSES"]}')
        print(f'Batch size: {param["batch"]}, learning rate: {param["LEARNING_RATE"]}')
        print(f'Augmentation: {param["augment"]}')
        print(f'Distance Map: {param["distance_map"]}')
        print('------------------------------------')
        if 'test_all' in param.keys():
            self.test_all = param['test_all']

        # setup class weights
        class_weights = torch.ones(self.n_classes)
        if self.n_classes == 4:
            class_weights = weights_5a
        elif self.n_classes == 9:
            class_weights = weights_5a9c
        elif self.n_classes == 10:
            class_weights = weights_5a10c

        # setup loss and training parameters
        # self.criterion = DiceLoss(weight=class_weights)
        self.criterion = DiceCELoss(class_weights=class_weights, nc=self.n_classes, ce_weight=1.0, dice_weight=0.5)
        # self.criterion = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.lr = param['LEARNING_RATE']
        self.kernel = param['kernel']

        # setup matrixes for loss, accuracy, dice score and f1 calculation
        self.val_loss_mat = torch.Tensor([])
        self.val_acc_mat = torch.Tensor([])
        self.val_f1_mat = torch.Tensor([])
        self.train_acc_mat = torch.Tensor([])
        self.train_f1_mat = torch.Tensor([])
        self.example_input_array = torch.rand((1, image_channels, 640, 384))        # 640, 384
        self.dice_score_total = torch.zeros(self.n_classes, device=self.device)
        self.mae_total = torch.zeros(self.n_classes-1, device=self.device)
        self.dice_score_table = torch.Tensor([], device=self.device)
        self.count_false = 0        # count falsly segmented preretinal space pixels (<80%)
        self.count_topology = 0     # parameter to count predictions with incorrect retina layers topology
        self.results = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1_score = self.shared_step(batch)
        self.log('train_loss', loss, batch_size=len(batch))
        self.train_acc_mat = torch.cat((self.train_acc_mat, torch.Tensor([accuracy])), dim=0)
        self.train_f1_mat = torch.cat((self.train_f1_mat, torch.Tensor([f1_score])), dim=0)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1_score = self.shared_step(batch)
        self.val_loss_mat = torch.cat((self.val_loss_mat, torch.Tensor([loss])), dim=0)
        self.val_acc_mat = torch.cat((self.val_acc_mat, torch.Tensor([accuracy])), dim=0)
        self.val_f1_mat = torch.cat((self.val_f1_mat, torch.Tensor([f1_score])), dim=0)
        self.log('val_loss', loss, batch_size=len(batch))
        if not self.trainer.sanity_checking:
            # save every 15th validation image to TensorBoard
            modulo_count = 15
            if (batch_idx % modulo_count) == 0:
                image = batch.get("image")
                gt = batch.get("mask")
                prediction = self.model(image)
                example = image_grid(image[0], gt[0], prediction[0])
                image_id = batch_idx / modulo_count
                self.logger.experiment.add_images("Example " + str(int(image_id + 1)), example, self.current_epoch,
                                                  dataformats='NCHW')
        return loss

    def shared_step(self, batch):
        image = batch.get("image")
        gt = batch.get("mask")
        pred = self.model(image)

        gt = torch.squeeze(gt, 1)
        pred = torch.squeeze(pred, 1)
        loss = self.criterion(pred, gt)
        accuracy, f1_score = calculate_metrics(pred, gt)
        return loss, accuracy, f1_score

    def test_step(self, batch, batch_idx):
        image = batch.get("image")
        gt = batch.get("mask")
        img_path = batch.get("image_path")
        path = ''
        if (self.create_dm is not None) and ('2Net' not in self.create_dm):
            prediction = None
        else:
            prediction = self.model(image)

        if image.size()[2] != 640:
            trans_resize_bilinear = transforms.Resize((640, 384), interpolation=func.InterpolationMode.BILINEAR)
            trans_resize_nearest = transforms.Resize((640, 384), interpolation=func.InterpolationMode.NEAREST)
            image = trans_resize_bilinear(image)
            gt = trans_resize_nearest(gt)
            prediction = trans_resize_nearest(prediction)

        # instead of testing the model, create Relative Distance Map and save it to file as image
        if self.create_dm is not None:
            if '2Net' in self.create_dm:
                th = 0.5
                pred = logit_to_one_hot(prediction, th)
                pred_cor = correct_prediction(pred)
            else:
                pred_cor = None
                image = image.cpu()
            distance_map = calculate_distance_map(image[0], method=self.create_dm, prediction=pred_cor)
            image_path = batch.get("image_path")[0].split(os.path.sep)
            image_path[0] = image_path[0] + '_DM' + self.create_dm + '_' + self.network + '_k' + str(self.kernel[0]) + str(self.kernel[1])
            os.makedirs(os.path.join(*image_path[:-1]), exist_ok=True)
            new_path = os.path.join(*image_path)
            io.imsave(new_path, distance_map.cpu().numpy())
            output = dict({})

        # otherwise: test the model
        else:
            th = 0.5
            pred = prediction > th
            acc = torch.Tensor([])
            dice_score = torch.Tensor([])
            class_area = torch.Tensor([])

            # calculate metrics between prediction and ground truth for each class
            for c in range(self.n_classes):
                fp, fn, tp, tn = numeric_score(pred[:, c, :, :].int(), gt[:, c, :, :])
                acc = torch.cat((acc, torch.Tensor([accuracy(fp, fn, tp, tn)])), dim=0)
                dice_score = torch.cat((dice_score, torch.Tensor([dice_coef(gt[:, c, :, :] > 0, pred[:, c, :, :])])), dim=0)
                class_area = torch.cat((class_area, torch.Tensor([torch.sum(gt[:, c, :, :])])), dim=0)
            if dice_score[1] < 0.8:     # get file name with low (<80%) dice score for second class (preretinal space)
                self.count_false += 1
                path = img_path
            self.dice_score_table = torch.cat((self.dice_score_table, torch.unsqueeze(dice_score, dim=1)), dim=1)

            # check correctness of retina topology
            topology = check_topology(pred)
            pred_cor = correct_prediction(pred)
            mae = mean_abs_error(gt, pred_cor)  # calculate MAE

            # save all results for this sample to a list
            result_list = [batch_idx]
            result_list.extend(acc.tolist())
            result_list.extend(dice_score.tolist())
            result_list.extend(class_area.tolist())
            result_list.extend(mae.tolist())
            result_list.extend([img_path[0]])
            is_error = 1 if (dice_score[0] < 0.8) or (dice_score[1] < 0.8) else 0
            result_list.extend([is_error])
            result_list.extend([topology])
            if topology == 1:
                self.count_topology += 1
            self.results.append(result_list)

            # either save every predicted image to TensorBoard
            if self.test_all:
                file_name = '/'.join(img_path[0].split(os.path.sep)[-2:])
                pred_sum = torch.squeeze(one_hot_to_sum(pred), 0) / 255
                # ax = plt.subplot()
                # ax.imshow(pred_sum.cpu(), vmin=0.25, vmax=1)
                # plt.show()
                self.logger.experiment.add_image("Test file " + str(int(batch_idx + 1)) + ": " + file_name, pred_sum,
                                                 dataformats='HW')
            # or only each 10th
            else:
                modulo_count = 10
                if (batch_idx % modulo_count) == 0:
                    image_id = batch_idx / modulo_count
                    file_name = '/'.join(img_path[0].split(os.path.sep)[-2:])
                    example = image_grid(image[0], gt[0], prediction[0])
                    self.logger.experiment.add_images("Test file " + str(int(image_id + 1)) + ": " + file_name, example, 0,
                                                      dataformats='NCHW')   # max_outputs=4

                # if image prediction has dice score lower than 80% for either first (top) or second (preretinal) class
                # - save to Tensorboard as erroneous
                if is_error:
                    file_name = '/'.join(img_path[0].split(os.path.sep)[-2:])
                    example = image_grid(image[0], gt[0], prediction[0])
                    self.logger.experiment.add_images("Error file: " + file_name, example, 0, dataformats='NCHW')

            print(f' Dice score: {dice_score}, MAE: {mae}, {path}')
            self.dice_score_total += dice_score
            self.mae_total += mae
            output = dict({
                'test_dice': dice_score,
                'test_mae': mae,
            })
        return output

    def training_epoch_end(self, outputs):
        # calculate average metrics for loss, accuracy and F1 score for a TRAINING epoch, save to TensorBoard
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        save_loss(os.path.join(self.model_path, "train_loss.csv"), self.current_epoch, avg_loss)
        self.logger.experiment.add_scalar("Train/Loss", avg_loss, self.current_epoch)

        avg_acc = self.train_acc_mat.mean()
        self.train_acc_mat = torch.Tensor([])
        self.logger.experiment.add_scalar("Train/Accuracy", avg_acc, self.current_epoch)

        avg_f1 = self.train_f1_mat.mean()
        self.train_f1_mat = torch.Tensor([])
        self.logger.experiment.add_scalar("Train/F1", avg_f1, self.current_epoch)

        # if self.current_epoch == 0:
        #     self.logger.experiment.add_graph(LitUnet(), self.example_input_array)

    def validation_epoch_end(self, outputs):
        # calculate average metrics for loss, accuracy and F1 score for a VALIDATION epoch, save to TensorBoard
        if not self.trainer.sanity_checking:
            avg_loss = self.val_loss_mat.mean()
            save_loss(os.path.join(self.model_path, "val_loss.csv"), self.current_epoch, avg_loss)  # save loss to file
            self.val_loss_mat = torch.Tensor([])
            self.logger.experiment.add_scalar("Validation/Loss", avg_loss, self.current_epoch)

            avg_acc = self.val_acc_mat.mean()
            self.val_acc_mat = torch.Tensor([])
            self.logger.experiment.add_scalar("Validation/Accuracy", avg_acc, self.current_epoch)

            avg_f1 = self.val_f1_mat.mean()
            self.val_f1_mat = torch.Tensor([])
            self.logger.experiment.add_scalar("Validation/F1", avg_f1, self.current_epoch)
            # print(f'Ended epoch {self.current_epoch}...')

    def test_epoch_end(self, outputs):
        # calculate average metrics for loss, accuracy and F1 score for a TEST epoch
        data_length = len(self.trainer.test_dataloaders[0])
        self.dice_score_total = self.dice_score_total / data_length
        self.mae_total = self.mae_total / data_length

        # print averaged results to console
        print(f'Dice score total: {self.dice_score_total}, MAE total: {self.mae_total}')
        print(f'Falsely segmented preretinal space (<80%): {self.count_false / data_length}')
        print(f'Incorrect topology: {self.count_topology / data_length}')

        # save to TensorBoard
        for i in range(len(self.dice_score_total)):
            self.logger.experiment.add_scalar("Test/Dice", self.dice_score_total[i], i)
        for i in range(len(self.mae_total)):
            self.logger.experiment.add_scalar("Test/MAE", self.mae_total[i], i)

        # save to .csv file
        columns = ['Idx']
        columns.extend(['Accuracy' + str(i + 1) for i in range(self.n_classes)])
        columns.extend(['Dice' + str(i + 1) for i in range(self.n_classes)])
        columns.extend(['Area' + str(i + 1) for i in range(self.n_classes)])
        columns.extend(['MAE' + str(i + 1) for i in range(self.n_classes-1)])
        columns.extend(['Image_path', 'Error', 'Topology'])
        feature_score = pandas.DataFrame(self.results, columns=columns)
        feature_score.to_csv(os.path.join(self.model_path, 'results.csv'), index=False)
        # plot_dice_boxplot(self.dice_score_table)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)     # weight_decay=0,
        return optimizer
