import warnings
warnings.filterwarnings("ignore")

import torch
import pytest

from train import UnsupervisedTrainer
from vit import ViTNoHead

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

