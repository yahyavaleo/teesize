import os
import torch
import numpy as np

from torchsummary import summary_string
from torch.utils.data import DataLoader

from tqdm import tqdm
from prettytable import PrettyTable

from lib.dataset import FashionDataset
from lib.model import Softmax2d
from lib.inference import get_landmarks


def get_correct_keypoints(prediction, ground_truth, threshold):
    distances = np.linalg.norm(prediction - ground_truth, axis=1)
    return (distances <= threshold).astype(int)


def compute_pck(root, model, threshold, savepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FashionDataset(root, num_landmarks=25, do_augment=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_keypoints = 25
    n = 0
    correct_keypoints = np.zeros(num_keypoints, dtype=int)

    print()
    print("Computing PCK ...")

    with torch.no_grad():
        for i, (image, target, meta) in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, target = image.to(device), target.to(device)

            output = model(image.float())
            output = Softmax2d(output)

            target = target.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            for prediction, ground_truth in zip(output, target):
                prediction = get_landmarks(prediction)
                ground_truth = get_landmarks(ground_truth)

                correct_keypoints += get_correct_keypoints(prediction, ground_truth, threshold)
                n += 1

    pck = correct_keypoints / n
    overall_pck = np.mean(pck)

    table = PrettyTable()
    table.field_names = ["Keypoint", "PCK"]

    for i, pck_i in enumerate(pck):
        table.add_row([f"Keypoint {i + 1}", f"{pck_i:.2%}"])

    print()
    print(table)
    print()
    print(f"Overall PCK: {overall_pck:.2%}")

    with open(savepath, "w") as f:
        f.write(str(table))
        f.write("\n")
        f.write(f"Overall PCK: {overall_pck:.2%}")


def model_summary(model, savepath):
    res, _ = summary_string(model, (1, 256, 256))
    print(res)

    with open(savepath, "w") as f:
        f.write(res)


def evaluate(dataset, model, threshold):
    output_dir = os.path.join("lib", "res")
    os.makedirs(output_dir, exist_ok=True)

    savepath = os.path.join(output_dir, "summary.txt")
    model_summary(model, savepath)

    savepath = os.path.join(output_dir, "pck.txt")
    compute_pck(dataset, model, threshold, savepath)
