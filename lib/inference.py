import torch
import numpy as np

import imgaug.augmenters as iaa
from skimage import io, img_as_ubyte
from timeit import default_timer as timer

from lib.model import Unet, Softmax2d


def transform(image):
    W = 256
    H = 256
    aspect_ratio = W / H

    preprocessing = iaa.Sequential(
        [
            iaa.PadToAspectRatio(aspect_ratio, position="right-bottom"),
            iaa.Resize({"width": W, "height": H}),
        ]
    )

    image = preprocessing(image=image)
    image = np.clip(image, 0.0, 1.0)
    image = img_as_ubyte(image)
    image = np.expand_dims(image, axis=(0, 1))
    image = torch.from_numpy(image)
    return image


def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = np.ndarray.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))


def get_landmarks(heatmaps):
    heatmaps = heatmaps / np.max(heatmaps, axis=(1, 2), keepdims=True)
    landmarks = np.array([get_hottest_point(heatmap) for heatmap in heatmaps])
    return landmarks


def predict(image_file, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = io.imread(image_file, as_gray=True)
    image = transform(image)
    image = image.to(device)

    with torch.no_grad():
        start = timer()

        output = model(image.float())
        output = Softmax2d(output)

        end = timer()
        inference_time = end - start

        image = image.detach().cpu().numpy()
        image = image[0, 0]

        output = output.detach().cpu().numpy()
        landmarks = get_landmarks(output[0])

        return image, landmarks, inference_time


def load_model(checkpoint_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet().to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return model
