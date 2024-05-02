import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            decoder_channels=[256, 128, 64, 32, 32],
            in_channels=1,
            classes=25,
        )

    def forward(self, x):
        return self.unet(x)


def Softmax2d(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def CrossEntropyLoss(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))
