import os
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from lib.dataset import FashionDataset
from lib.model import Unet, Softmax2d, CrossEntropyLoss


def train(root, checkpoint_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["epoch"]

    lr_scheduler = LambdaLR(optimizer, lr_schedule, last_epoch=last_epoch)

    dataset = FashionDataset(root, num_landmarks=25, do_augment=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    epochs = 50
    for epoch in range(last_epoch, epochs):
        model.train()

        for i, (image, target, meta) in enumerate(dataloader):
            image, target = image.to(device), target.to(device)

            output = model(image.float())
            output = Softmax2d(output)

            optimizer.zero_grad()
            loss = CrossEntropyLoss(output, target)

            optimizer.step()

            if i % 5 == 0:
                log(epoch, epochs, i, len(dataloader), loss.item(), lr_scheduler.get_last_lr())

        output_dir = os.path.join("lib", "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        savepath = os.path.join(output_dir, f"epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, savepath)

        lr_scheduler.step()


def lr_schedule(epoch):
    if 10 < epoch < 20:
        return 1e-3
    elif 20 < epoch < 30:
        return 1e-2
    elif 30 < epoch:
        return 1e-3


def save_checkpoint(model, optimizer, epoch, savepath):
    states = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
    }
    torch.save(states, savepath)


def log(epoch, num_epochs, i, count, loss, lr):
    print(f"Epoch: {epoch}/{num_epochs} [{i}/{count}]\tLoss: {loss:.5f} LR: {lr}")
