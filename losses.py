import torch
import torch.nn as nn
import math
from skimage.feature import canny
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
import timeit


def per_class_dice_score(prediction, target, class_weight, smooth, no_class=None):
    pred = prediction.view(-1)
    true = target.view(-1)
    if no_class is not None and torch.sum(no_class) > 0:
        nc = no_class.view(-1)
        pred = torch.masked_select(pred, ~nc)
        true = torch.masked_select(true, ~nc)
    intersection = torch.dot(pred, true.to(torch.float32))
    numerator = 2. * class_weight * intersection + smooth
    denumerator = torch.add(torch.sum(pred), torch.sum(true)) + smooth        # torch.square()

    if math.isnan(numerator):
        print(f'Numerator is NAN!')
    if math.isnan(denumerator):
        print(f'Denumerator is NAN!')
    elif denumerator == 0:
        print(f'Denumerator is Zero!')

    return torch.divide(numerator, denumerator)


class DiceLoss(nn.Module):

    def __init__(self, weight=None, nc=4):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.n_classes = nc

    def forward(self, y_pred, y_true, smooth=1.0):
        assert y_pred.size() == y_true.size()
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        if self.weight is None:
            self.weight = [1.0, 1.0, 1.0, 1.0]

        dice = 0.
        if self.n_classes == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.contiguous().view(-1)
            y_true = y_true.contiguous().view(-1)
            intersection = (y_pred * y_true).sum()
            dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        else:
            activation_fn = nn.Softmax2d()
            y_pred = activation_fn(y_pred)
            y_pred = y_pred.contiguous()
            y_true = y_true.contiguous()
            for c in range(self.n_classes):
                dice += per_class_dice_score(y_pred[:, c, :, :], y_true[:, c, :, :], self.weight[c], smooth)
        loss = 1. - dice
        return loss


class DiceCELoss(nn.Module):
    def __init__(self, class_weights, nc=4, ce_weight=1.0, dice_weight=1.0, edge_weight=10, area_weights=(1, 5, 5, 1)):
        super(DiceCELoss, self).__init__()
        self.class_weights = class_weights
        self.n_classes = nc
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        if len(area_weights) != nc:
            self.area_weights = (1,)
            self.area_weights += (5,) * (nc - 2)
            self.area_weights += (1,)
        else:
            self.area_weights = area_weights
        self.ce_mat_weights_all = torch.Tensor([])
        self.no_class_pixels = torch.Tensor([])
        self.epsilon = 1e-15

    def forward(self, y_pred, y_true, smooth=1):
        activation_fn = nn.Softmax2d()
        y_pred = activation_fn(y_pred)
        dice = 0.
        ce_loss = 0.

        self.ce_mat_weights_all = torch.zeros_like(y_true[:, 0, :, :])
        self.no_class_pixels = torch.sum(y_true, dim=1) == 0
        for c in range(self.n_classes):
            pred = y_pred[:, c, :, :].contiguous()
            true = y_true[:, c, :, :].contiguous()
            dice += per_class_dice_score(pred, true, self.class_weights[c], smooth, self.no_class_pixels)
            if math.isnan(dice):
                print('Dice loss is NAN!')

            true = true.to(torch.int8)
            batch_size = true.size()[0]
            empty_row = torch.zeros(batch_size, 1, true.size()[2]).cuda()
            edges_top = torch.sub(true[:, 1:, :], true[:, :-1, :]).clamp(0, 1)
            edges_top = torch.cat((empty_row, edges_top), dim=1)
            edges_bottom = torch.sub(true[:, :-1, :], true[:, 1:, :]).clamp(0, 1)
            edges_bottom = torch.cat((edges_bottom, empty_row), dim=1)
            edges = edges_top + edges_bottom

            true_no_edges = true.clone()
            true_no_edges[torch.gt(edges, 0)] = 0
            matrix = torch.add(torch.mul(torch.gt(edges, 0), self.edge_weight), torch.mul(true_no_edges, self.area_weights[c]))
            self.ce_mat_weights_all.add_(matrix)

        for c in range(self.n_classes):
            pred_ce = y_pred[:, c, :, :].contiguous().clamp(self.epsilon)
            true_ce = y_true[:, c, :, :].contiguous()
            if torch.sum(self.no_class_pixels) > 0:
                weight_matrix = torch.masked_select(self.ce_mat_weights_all, ~self.no_class_pixels)
                pred_ce = torch.masked_select(pred_ce, ~self.no_class_pixels)
                true_ce = torch.masked_select(true_ce, ~self.no_class_pixels)
            else:
                weight_matrix = self.ce_mat_weights_all
            ce_loss -= torch.mul(torch.mul(weight_matrix, true_ce), torch.log(pred_ce)).mean()
            # second_term = (1 - true_ce) * torch.log(1 - pred_ce)

        if math.isnan(ce_loss):
            print('Cross entropy loss is NAN!')

        dice_ce_loss = torch.add(torch.mul(self.ce_weight, ce_loss), torch.mul(self.dice_weight, (1 - dice)))
        return dice_ce_loss


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, class_weights=None, nc=4):
        super(BCEWithLogitsLoss, self).__init__()
        if class_weights is None:
            self.class_weights = torch.ones(nc)
        else:
            self.class_weights = class_weights
        self.n_classes = nc
        self.no_class_pixels = torch.Tensor([])
        self.epsilon = 1e-15

    def forward(self, y_pred, y_true):
        activation_fn = nn.Sigmoid()
        y_pred = activation_fn(y_pred)
        bce_loss = 0.
        self.no_class_pixels = torch.sum(y_true, dim=1) == 0

        for c in range(self.n_classes):
            pred = y_pred[:, c, :, :].contiguous().clamp(self.epsilon, 0.999)
            true = y_true[:, c, :, :].contiguous()
            if torch.sum(self.no_class_pixels) > 0:
                pred = torch.masked_select(pred, ~self.no_class_pixels)
                true = torch.masked_select(true, ~self.no_class_pixels)
            pos = torch.mul(torch.mul(self.class_weights[c], true), torch.log(pred))
            neg = torch.mul((1 - true), torch.log(1 - pred))
            bce_loss -= torch.mean(torch.add(pos, neg))
            if math.isnan(bce_loss):
                pred_ones = torch.sum(pred == 1.0)
                if pred_ones:
                    print(f'Pred >= 1! {pred_ones}')
                print(f'Class {c}: pos - {torch.mean(pos)}, neg - {torch.mean(neg)}')

        return bce_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, nc=4):
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is None:
            self.class_weights = torch.Tensor([1.0, 1.0, 1.0, 1.0])
        else:
            self.class_weights = class_weights
        self.n_classes = nc

    def forward(self, y_pred, y_true):
        activation_fn = nn.Softmax2d()
        y_pred = activation_fn(y_pred)
        ce_loss = 0.
        for c in range(self.n_classes):
            pred = y_pred[:, c, :, :].contiguous()  # .view(-1)
            true = y_true[:, c, :, :].contiguous()  # .view(-1)
            ce_loss += self.class_weights[c] * (- true * torch.log(pred)).mean()

        return ce_loss
