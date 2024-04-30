import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


def visualize(images_dir, annotations_dir, idx):
    image_file = os.path.join(images_dir, f"{idx:06d}.jpg")
    image = Image.open(image_file)

    annotations_file = os.path.join(annotations_dir, f"{idx:06d}.json")
    with open(annotations_file, "r") as f:
        data = json.load(f)

    landmarks = data["landmarks"]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    points = np.array(landmarks)
    ax.plot(
        points[[0, 1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1, 0, 5], 0],
        points[[0, 1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1, 0, 5], 1],
        color="#9146FF",
        linewidth=2,
        linestyle="-",
        marker="s",
        markersize=4,
        markeredgecolor="black",
    )

    poly = patches.Polygon(
        points[[1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1], :2],
        closed=True,
        facecolor="#9146FF",
        alpha=0.3,
    )
    ax.add_patch(poly)

    plt.title(f"Shirt #{idx}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <idx>")
        sys.exit(1)

    images_dir = os.path.join("shirts", "train", "images")
    annotations_dir = os.path.join("shirts", "train", "annotations")

    idx = int(sys.argv[1])
    visualize(images_dir, annotations_dir, idx)
