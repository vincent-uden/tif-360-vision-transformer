import warnings
warnings.filterwarnings("ignore")

import torch
import pytest
import numpy as np
import scipy

from train import UnsupervisedTrainer
from vit import ViTNoHead
from matplotlib import pyplot as plt
from gen_data import GeneratedDataset
from torch.utils.data.dataloader import DataLoader

def test_trainer_loss():
    model = ViTNoHead(
        colors=3,
        height=16,
        width=16,
        n_patches=2,
        hidden_dimension=4,
        n_heads=1,
        n_blocks=1
    )
    head = torch.nn.Sequential(
        torch.nn.Linear(4, 2),
        torch.nn.Softmax(dim=-1),
    )
    trainer = UnsupervisedTrainer(model, head, "cpu")

    softmax = torch.nn.Softmax(dim=-1)
    test_cls = softmax(torch.tensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0],
    ], dtype=torch.float32))
    test_target = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
    ], dtype=torch.float32)

    assert test_cls.shape[0] == 4 and test_cls.shape[1] == 3

    loss = trainer.loss(test_cls, test_target)

    assert torch.all(loss > 0), "A loss should always be positive"

def interp_att_map(att: np.ndarray) -> np.ndarray:
    X = np.linspace(0, 256, 16)
    Y = np.linspace(0, 256, 16)
    x,y = np.meshgrid(X, Y)
    print(att.shape)
    print(X, Y)

    f = scipy.interpolate.RectBivariateSpline(X, Y, att)

    Xnew = np.linspace(0, 256, 256)
    Ynew = np.linspace(0, 256, 256)

    return f(Xnew, Ynew)

def test_attention_map(show_att=False):
    dims = 256
    patches = 16
    model = ViTNoHead(
        colors=3,
        height=dims,
        width=dims,
        n_patches=patches,
        hidden_dimension=32,
        n_heads=8,
        n_blocks=10
    )
    model.load_state_dict(torch.load("vitnohead_shuffle2.pt", map_location=torch.device("cpu")))

    test_data = GeneratedDataset(128, start_i=256)
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=True
    )

    test_img, _ = next(iter(test_loader))
    print(test_img.shape)
    att = interp_att_map(model.attention_rollout(test_img))
    att = att / np.max(att)
    att_mean = interp_att_map(model.attention_rollout(test_img, np.mean))
    att_mean = att_mean / np.max(att_mean)
    att_min = interp_att_map(model.attention_rollout(test_img, np.min))
    att_min = att_min / np.max(att_min)

    if show_att:
        plt.subplot(2, 2, 1)
        plt.imshow(test_img[0].permute(1, 2, 0))
        plt.title("Source Image")
        plt.subplot(2, 2, 2)
        plt.imshow(test_img[0].permute(1, 2, 0) * np.repeat(att[:,:,np.newaxis], 3, axis=2))
        plt.title("Max Attention")
        plt.subplot(2, 2, 3)
        plt.imshow(test_img[0].permute(1, 2, 0) * np.repeat(att_mean[:,:,np.newaxis], 3, axis=2))
        plt.title("Mean Attention")
        plt.subplot(2, 2, 4)
        plt.imshow(test_img[0].permute(1, 2, 0) * np.repeat(att_min[:,:,np.newaxis], 3, axis=2))
        plt.title("Min Attention")
        plt.show()

def test_image_gen(show=False):
    dataset = GeneratedDataset(1048)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    xs1, ys1 = [], []
    xs2, ys2 = [], []

    for i, batch in enumerate(loader):
        x, y = batch
        if i == 10:
            break

    if show:
        imgs = torch.permute(x, (0, 2, 3, 1))
        plt.subplot(1, 4, 1)
        plt.imshow(imgs[0])
        plt.subplot(1, 4, 2)
        plt.imshow(imgs[1])
        plt.subplot(1, 4, 3)
        plt.imshow(imgs[2])
        plt.subplot(1, 4, 4)
        plt.imshow(imgs[3])

        plt.show()

    for (x, y) in loader:
        xs1.append(x)
        ys1.append(y)

    for (x, y) in loader:
        xs2.append(x)
        ys2.append(y)

    for (x1, x2) in zip(xs1, xs2):
        assert (x1 == x2).all()

    for (y1, y2) in zip(ys1, ys2):
        assert (y1 == y2).all()


