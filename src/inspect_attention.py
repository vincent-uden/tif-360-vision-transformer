import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from vit import ViTNoHead
from gen_data import GeneratedDataset
from tqdm import tqdm, trange


if __name__ == "__main__":
    model = ViTNoHead(
        colors=3,
        height=256,
        width=256,
        n_patches=8,
        hidden_dimension=8,
        n_heads=1,
        n_blocks=1
    )

    model.load_state_dict(torch.load("first_vit.pt"))

    test_data = GeneratedDataset(1048, start_i=8192)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    model.eval()

    for (x, y) in test_loader:
        for i in range(4):
            print(x[i:i+1].shape)
            att = model.attention_rollout(x[i:i+1])

            plt.subplot(2, 4, i+1)
            plt.imshow(x[i].permute(1, 2, 0))
            plt.subplot(2, 4, 4+i+1)
            plt.imshow(att)
        plt.show()
