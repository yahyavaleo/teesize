import warnings
import torch.ao.quantization
from torch.jit import TracerWarning

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=TracerWarning)

    model = Unet()
    checkpoint = torch.load("weights/epoch_50.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    example_input = torch.rand(4, 1, 256, 256)

    # ---------------------------------
    # Post-Training Static Quantization
    # ---------------------------------
    # model.eval()
    # model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    # quantized_model = torch.quantization.prepare(model.unet)
    #
    # with torch.no_grad():
    #     model(example_input)
    #
    # quantized_model = torch.quantization.convert(model.unet)

    # ----------------------------------
    # Post-Training Dynamic Quantization
    # ----------------------------------
    # quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Conv2d}, dtype=torch.qint8)

    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("resnet.pt")

        print("Traced model saved!")
    except Exception:
        print("There was an error trying to convert the model to Torch Script!")

    try:
        onnx_model = torch.onnx.export(
            model, (example_input), "resnet.onnx", input_names=["input"], output_names=["output"]
        )

        print("ONNX export successful!")
    except Exception:
        print("There was an error trying to export the model to ONNX format!")
