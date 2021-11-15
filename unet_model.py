import torch
import torch.nn as nn
import torchvision.models
import numpy as np


class BasicUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64, conv_ker=(3, 3)):
        super(BasicUNet, self).__init__()

        conv_pad = (int((conv_ker[0] - 1) / 2), int((conv_ker[1] - 1) / 2))
        features = init_features
        self.encoder1 = BasicUNet._block(in_channels, features, name="enc1", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder2 = BasicUNet._block(features, features * 2, name="enc2", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder3 = BasicUNet._block(features * 2, features * 4, name="enc3", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder4 = BasicUNet._block(features * 4, features * 8, name="enc4", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck = BasicUNet._block(features * 8, features * 16, name="bottleneck", conv_kernel=conv_ker, conv_pad=conv_pad)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=(2, 2), stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = BasicUNet._block((features * 8) * 2, features * 8, name="dec4", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(2, 2), stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = BasicUNet._block((features * 4) * 2, features * 4, name="dec3", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = BasicUNet._block((features * 2) * 2, features * 2, name="dec2", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = BasicUNet._block(features * 2, features, name="dec1", conv_kernel=conv_ker, conv_pad=conv_pad)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=(1, 1))

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4, indices4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        return enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck

    def forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    def forward(self, x):
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck = BasicUNet.forward_encoder(self, x)
        dec1 = BasicUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, conv_kernel, conv_pad, batch_norm=False, repetitions=2):
        block_sequence = nn.Sequential()
        for count in range(repetitions):
            block_sequence.add_module(name + "conv" + str(count+1), nn.Conv2d(in_channels=in_channels,
                                                                            out_channels=features,
                                                                            kernel_size=conv_kernel,
                                                                            padding=conv_pad, bias=False))
            if batch_norm:
                block_sequence.add_module(name + "norm" + str(count+1), nn.BatchNorm2d(num_features=features))
            block_sequence.add_module(name + "relu" + str(count+1), nn.ReLU(inplace=True))
            in_channels = features
        return block_sequence



class UNetOrg(BasicUNet):
    def __init__(self, params):
        super().__init__(in_channels=params['in_channels'], out_channels=params['N_CLASSES'],
                         init_features=params['features'], conv_ker=params['kernel'])


class ReLayNet(BasicUNet):
    def __init__(self, params):
        if 'in_channels' not in params:
            params['in_channels'] = 3
        if 'N_CLASSES' not in params:
            params['N_CLASSES'] = 1
        if 'features' not in params:
            params['features'] = 64
        if 'kernel' not in params:
            params['kernel'] = (7, 3)
        super().__init__(params['in_channels'], params['N_CLASSES'], params['features'], params['kernel'])

        conv_pad = (int((params['kernel'][0]-1)/2), int((params['kernel'][1]-1)/2))
        features = params['features']
        kernel = params['kernel']
        self.encoder1 = ReLayNet._block(params['in_channels'], features, name="enc1", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)
        self.encoder2 = ReLayNet._block(features, features, name="enc2", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)
        self.encoder3 = ReLayNet._block(features, features, name="enc3", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)

        self.bottleneck = ReLayNet._block(features, features, name="bottleneck", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)

        self.decoder3 = ReLayNet._block(features * 2, features, name="dec3", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)
        self.decoder2 = ReLayNet._block(features * 2, features, name="dec2", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)
        self.decoder1 = ReLayNet._block(features * 2, features, name="dec1", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=True, repetitions=1)

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        bottleneck = self.bottleneck(pool3)
        return enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck

    def forward(self, x):
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck = ReLayNet.forward_encoder(self, x)

        dec3 = self.unpool3(bottleneck, indices3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.unpool2(dec3, indices2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.unpool1(dec2, indices1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


class LFUNet(BasicUNet):

    def __init__(self, params, noU=False):
        super().__init__(in_channels=params['in_channels'], out_channels=params['N_CLASSES'],
                         init_features=params['features'], conv_ker=params['kernel'])

        conv_pad = (int((params['kernel'][0]-1)/2), int((params['kernel'][1]-1)/2))
        features = params['features']

        self.upconv4a = nn.ConvTranspose2d(features * 16, features * 4, kernel_size=(2, 2), stride=2)
        self.upconv4b = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=2)
        self.upconv4c = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=2)
        self.upconv4d = nn.ConvTranspose2d(features, features, kernel_size=(2, 2), stride=2)

        dilation4 = (params['kernel'][0]+1, params['kernel'][1]+1)
        dilation6 = (params['kernel'][0]+3, params['kernel'][1]+3)
        dilation8 = (params['kernel'][0]+5, params['kernel'][1]+5)
        paddil4 = (conv_pad[0] * dilation4[0], conv_pad[1] * dilation4[1])
        paddil6 = (conv_pad[0] * dilation6[0], conv_pad[1] * dilation6[1])
        paddil8 = (conv_pad[0] * dilation8[0], conv_pad[1] * dilation8[1])
        self.convdil4 = nn.Conv2d(features * 2, int(features / 2), kernel_size=params['kernel'], padding=paddil4, bias=False, dilation=dilation4)
        self.convdil6 = nn.Conv2d(features * 2, int(features / 2), kernel_size=params['kernel'], padding=paddil6, bias=False, dilation=dilation6)
        self.convdil8 = nn.Conv2d(features * 2, int(features / 2), kernel_size=params['kernel'], padding=paddil8, bias=False, dilation=dilation8)

        self.dropout = nn.Dropout2d()
        self.conv = nn.Conv2d(in_channels=int(features / 2 * 3), out_channels=params['N_CLASSES'], kernel_size=(1, 1))

        # self.noUnet = noU
        # self.convNoUnet = nn.Conv2d(features, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck = LFUNet.forward_encoder(self, x)
        dec1 = LFUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)

        fcn4 = self.upconv4a(bottleneck) + pool3
        fcn3 = self.upconv4b(fcn4) + pool2
        fcn2 = self.upconv4c(fcn3) + pool1
        fcn1 = self.upconv4d(fcn2)

        # if self.noUnet:
        #     return self.convNoUnet(fcn1)
        # else:
        fcn = torch.cat((dec1, fcn1), dim=1)
        dil8 = self.convdil8(fcn)
        dil6 = self.convdil6(fcn)
        dil4 = self.convdil4(fcn)

        last = torch.cat((dil8, dil6, dil4), dim=1)
        drop = self.dropout(last)

        return self.conv(drop)



class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1)),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1)),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttUNet(BasicUNet):
    def __init__(self, params):
        super().__init__(in_channels=params['in_channels'], out_channels=params['N_CLASSES'],
                         init_features=params['features'], conv_ker=params['kernel'])

        conv_pad = (int((params['kernel'][0] - 1) / 2), int((params['kernel'][1] - 1) / 2))
        features = params['features']
        kernel = params['kernel']
        bn = True       # use batch normalization

        self.encoder1 = AttUNet._block(params['in_channels'], features, name="enc1", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.encoder2 = AttUNet._block(features, features * 2, name="enc2", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.encoder3 = AttUNet._block(features * 2, features * 4, name="enc3", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.encoder4 = AttUNet._block(features * 4, features * 8, name="enc4", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)

        self.bottleneck = MyUNet._block(features * 8, features * 16, name="bottleneck", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)

        self.decoder4 = AttUNet._block((features * 8) * 2, features * 8, name="dec4", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.decoder3 = AttUNet._block((features * 4) * 2, features * 4, name="dec3", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.decoder2 = AttUNet._block((features * 2) * 2, features * 2, name="dec2", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)
        self.decoder1 = AttUNet._block(features * 2, features, name="dec1", conv_kernel=kernel, conv_pad=conv_pad, batch_norm=bn)

        self.att5 = AttentionBlock(F_g=(features * 8), F_l=(features * 8), F_int=(features * 4))
        self.att4 = AttentionBlock(F_g=(features * 4), F_l=(features * 4), F_int=(features * 2))
        self.att3 = AttentionBlock(F_g=(features * 2), F_l=(features * 2), F_int=features)
        self.att2 = AttentionBlock(F_g=features, F_l=features, F_int=int(features / 2))

        self.Up5 = AttUNet._up_conv(ch_in=(features * 16), ch_out=(features * 8), conv_kernel=kernel, conv_pad=conv_pad)
        self.Up4 = AttUNet._up_conv(ch_in=(features * 8), ch_out=(features * 4), conv_kernel=kernel, conv_pad=conv_pad)
        self.Up3 = AttUNet._up_conv(ch_in=(features * 4), ch_out=(features * 2), conv_kernel=kernel, conv_pad=conv_pad)
        self.Up2 = AttUNet._up_conv(ch_in=(features * 2), ch_out=features, conv_kernel=kernel, conv_pad=conv_pad)

    def forward(self, x):
        # encoding path
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck = AttUNet.forward_encoder(self, x)

        # decoding + concat path
        dec4 = self.Up5(bottleneck)
        enc4 = self.att5(g=dec4, x=enc4)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.Up4(dec4)
        enc3 = self.att4(g=dec3, x=enc3)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.Up3(dec3)
        enc2 = self.att3(g=dec2, x=enc2)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.Up2(dec2)
        enc1 = self.att2(g=dec1, x=enc1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.conv(dec1)
        return output

    @staticmethod
    def _up_conv(ch_in, ch_out, conv_kernel, conv_pad):
        block_sequence = nn.Sequential()
        block_sequence.add_module("upsamp", nn.Upsample(scale_factor=2))
        block_sequence.add_module("conv2d", nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=conv_kernel,
                                                      padding=conv_pad, bias=False))
        block_sequence.add_module("norm", nn.BatchNorm2d(num_features=ch_out))
        block_sequence.add_module("relu", nn.ReLU(inplace=True))
        return block_sequence
