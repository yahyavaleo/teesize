import cv2
import numpy as np

from matplotlib import patches
import matplotlib.pyplot as plt


def draw_landmarks(image, landmarks, savepath):
    fig, ax = plt.subplots(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    plt.savefig(savepath)
    plt.close(fig)
