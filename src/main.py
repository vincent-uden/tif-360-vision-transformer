import numpy as np
import torch

from datasets import load_dataset, Dataset
from matplotlib import pyplot as plt
from vit import ViT
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

def img_to_tensor(row):
    row["image"] = torch.tensor(np.asarray(row["image"])).permute(2, 0, 1).to(torch.float32)
    return row

def obj_to_float(row):
    if len(row["objects"]["area"]) > 0:
        row["label"] = 1.0
    else:
        row["label"] = 0.0
    return row

def weighted_mse_loss(inp, target, weight):
    return (weight * (inp - target) ** 2).mean()

if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    model = ViT(3, 416, 416, 8, 512, 16, 5, 1).to(device)

    train_data = load_dataset("keremberke/csgo-object-detection", "full", split="train")
    test_data = load_dataset("keremberke/csgo-object-detection", "full", split="test")

    train_labels = [ float(len(row["objects"]["area"]) > 0) for row in train_data ]
    train_data = train_data.add_column("label", train_labels)
    train_data.set_format(type="torch", columns=["image", "label"])

    test_labels = [ float(len(row["objects"]["area"]) > 0) for row in test_data ]
    test_data = test_data.add_column("label", test_labels)
    test_data.set_format(type="torch", columns=["image", "label"])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=64)

    print(len(train_data), len(test_data))
    # print(next(iter(train_loader)))

    EPOCHS = 15
    LR = 0.00001

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    for epoch in trange(EPOCHS, desc="Training"):
        train_loss = 0.0
        correct, total = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x_name, y_name = batch
            x = batch[x_name].to(torch.float32) / 255.0
            y = batch[y_name]
            x, y = x.permute(0, 3, 1, 2).to(device), torch.unsqueeze(y.to(device), -1).to(torch.float32)
            y_hat = model(x)
            weights = ((y == 1.0) * 0.1 + (y == 0.0) * 0.9).detach()
            # loss = criterion(y_hat, y)
            loss = weighted_mse_loss(y_hat, y, weights)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            total += len(x)
            correct += (torch.round(y_hat) == y).detach().cpu().sum()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} loss: {train_loss:.12f} Training acc: {correct/total:.12f}")

    model.eval()
    correct, total, correct_1, correct_0 = 0, 0, 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x_name, y_name = batch
        x = batch[x_name].to(torch.float32) / 255.0
        y = batch[y_name]
        x, y = x.permute(0, 3, 1, 2).to(device), torch.unsqueeze(y.to(device), -1).to(torch.float32)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        test_loss += loss.detach().cpu().item() / len(train_loader)
        total += len(x)
        correct_mask = (torch.round(y_hat) == y).detach().cpu()
        correct += correct_mask.sum()
        correct_0 += torch.logical_and(correct_mask,(y == 0.0)).sum()
        correct_1 += torch.logical_and(correct_mask,(y == 1.0)).sum()

        del x
        del y
        del y_hat

    print(f"Test loss: {test_loss}")
    print(f"Test Accuracy: {correct / total:.12f}  {correct} {total} {correct_0} {correct_1}")

