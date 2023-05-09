import warnings
warnings.filterwarnings("ignore")

import torch
import pytest

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

def test_attention_map(show_att=False):
    dims = 256
    patches = 16
    model = ViTNoHead(
        colors=3,
        height=dims,
        width=dims,
        n_patches=patches,
        hidden_dimension=4,
        n_heads=1,
        n_blocks=3
    )

    test_img = torch.rand(1, 3, dims, dims)
    att = model.attention_rollout(test_img)

    assert att.shape == (patches, patches)

    if show_att:
        plt.imshow(att)
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


