import os
import json
import skimage

import numpy as np
import imgaug.augmenters as iaa

from tqdm import tqdm
from skimage import io
from torch.utils.data import Dataset
from imgaug.augmentables import KeypointsOnImage


class FashionDataset(Dataset):
    def __init__(self, root, num_landmarks, do_augment):
        self.root = os.path.normpath(root)
        self.num_landmarks = num_landmarks
        self.do_augment = do_augment
        self.db = self.load()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        record = self.db[idx]

        image_file = record["image_file"]
        landmarks = record["landmarks"]

        image = io.imread(image_file, as_gray=True)
        landmarks = np.array([[point[0], point[1]] for point in landmarks])

        image, landmarks = self.transform(image, landmarks)

        if self.do_augment:
            image, landmarks = self.augment(image, landmarks)

        target = self.heatmaps(image, landmarks)
        image = np.expand_dims(image, axis=0)

        w = record["width"] if "width" in record else 0
        h = record["height"] if "height" in record else 0
        s = record["scale"] if "scale" in record else 0
        o = record["occlusion"] if "occlusion" in record else 0
        z = record["zoom"] if "zoom" in record else 0
        v = record["viewpoint"] if "viewpoint" in record else 0

        meta = {"width": w, "height": h, "scale": s, "occlusion": o, "zoom": z, "viewpoint": v}
        return image, target, meta

    def transform(self, image, landmarks):
        W = 256
        H = 256
        aspect_ratio = W / H

        preprocessing = iaa.Sequential(
            [
                iaa.PadToAspectRatio(aspect_ratio, position="right-bottom"),
                iaa.Resize({"width": W, "height": H}),
            ]
        )

        landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=image.shape)
        image, landmarks = preprocessing(image=image, keypoints=landmarks)
        landmarks = landmarks.to_xy_array()

        image = np.clip(image, 0.0, 1.0)
        image = skimage.img_as_ubyte(image)

        return image, landmarks

    def augment(self, image, landmarks):
        augmentation = iaa.Sequential(
            [
                iaa.Affine(
                    translate_px={"x": (-50, 50), "y": (-50, 50)},
                    scale=[0.8, 1],
                    rotate=[-15, 15],
                )
            ]
        )

        landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=image.shape)
        image, landmarks = augmentation(image=image, keypoints=landmarks)
        landmarks = landmarks.to_xy_array()

        return image, landmarks

    def heatmaps(self, image, landmarks):
        target = np.zeros([self.num_landmarks, image.shape[0], image.shape[1]])

        for i, (x, y) in enumerate(landmarks):
            x = int(min(x, image.shape[1] - 1))
            y = int(min(y, image.shape[0] - 1))
            target[i, y, x] = 1.0

        return target

    def load(self):
        tmp_db = []
        count = len(os.listdir(os.path.join(self.root, "images")))

        print(f"Creating {os.path.basename(self.root)} database ...")
        for idx in tqdm(range(count)):
            idx = idx + 1
            filename = f"{idx:06d}"
            image_file = os.path.join(self.root, "images", filename + ".jpg")
            annotation_file = os.path.join(self.root, "annotations", filename + ".json")

            with open(annotation_file, "r") as f:
                a = json.load(f)

            landmarks = np.array(a["landmarks"])
            width = np.array(a["width"])
            height = np.array(a["height"])
            scale = np.array(a["scale"])
            occlusion = np.array(a["occlusion"])
            zoom = np.array(a["zoom"])
            viewpoint = np.array(a["viewpoint"])

            tmp_db.append(
                {
                    "image_file": image_file,
                    "landmarks": landmarks,
                    "width": width,
                    "height": height,
                    "scale": scale,
                    "occlusion": occlusion,
                    "zoom": zoom,
                    "viewpoint": viewpoint,
                }
            )
        return tmp_db
