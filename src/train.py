import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from vit import ViTNoHead

# [/] Create ViT With Separate Classifier
# [ ] Implement a trainer class for unsupervised learning
#     [/] Data Loading
#     [x] Custom Cross Entropy Loss
#     [x] Measure Accuracy
#     [x] Visualisation of Training
#     [x] Train Latent Space Classifier (after ViT)
# [x] Extract Attention Map


class UnsupervisedTrainer:
    def __init__(
        self,
        vit: ViTNoHead,
        head: torch.nn.Module,
        device: torch._C.device,
        lr=0.005,
        vit_epochs=5,
        head_epochs=5,
        batch_size=4,
    ):
        self.vit = vit
        self.head = head

        self.vit_opt = torch.optim.Adam(self.vit.parameters(), lr=lr)
        self.head_opt = torch.optim.Adam(self.head.parameters(), lr=lr)

        self.head_loss = torch.nn.BCELoss()

        self.batch_size = batch_size
        self.vit_epochs = vit_epochs
        self.head_epochs = head_epochs
        self.device = device
        assert self.batch_size % 4 == 0, "Batch size must be divisible by 4"

        self.train_data = []
        self.test_data = []

        self.train_loader = DataLoader(self.train_data, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=4, shuffle=False)

        # Stats
        self.vit_train_loss_per_epoch = np.zeros(self.vit_epochs)
        self.vit_test_loss_per_epoch = np.zeros(self.vit_epochs)

        self.head_train_loss_per_epoch = np.zeros(self.head_epochs)
        self.head_test_loss_per_epoch = np.zeros(self.head_epochs)

    def loss(self, cls_token: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # I expect the label to be on the form
        #   [[1, 0]]: Waldo
        #   [[0, 1]]: Not Waldo
        # with dimension (batch_size, 2)

        assert (
            cls_token.shape[0] % 4 == 0
        ), "Network output batch must be divisible by 4"
        assert (
            label.shape[0] % 4 == 0 and label.shape[1] > 0
        ), "Target batch must be divisible by 4"

        output_similarities = torch.mm(cls_token, torch.transpose(cls_token, 0, 1))
        target_similarities = torch.mm(label, torch.transpose(label, 0, 1))

        categorical_loss = -torch.sum(
            (torch.mul(target_similarities, torch.log(output_similarities)))
        )

        return categorical_loss

    def accuracy(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # I expect the label to be on the form
        #   [[1, 0]]: Waldo
        #   [[0, 1]]: Not Waldo
        # with dimension (batch_size, 2)

        cls_token = self.vit(data)
        pred = self.head(cls_token)

        correct_by_category = torch.sum(pred == labels, dim=0)

        return correct_by_category

    def train(self):
        epoch_pbar = trange(self.vit_epochs)

        for epoch in epoch_pbar:
            batch_pbar = tqdm(self.train_loader, leave=False)

            self.vit.train()
            b = 0
            for batch in batch_pbar:
                self.vit_opt.zero_grad()

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                cls_token = self.vit(x)
                loss = self.loss(cls_token, y)

                loss_float = loss.detach().cpu().item()
                self.vit_train_loss_per_epoch[epoch] += loss_float / len(
                    self.train_loader
                )

                loss.backward()
                self.vit_opt.step()

                batch_pbar.set_description(
                    f"Batch {b}/{len(self.train_loader)}, Loss {loss_float/self.batch_size}"
                )

            epoch_pbar.set_description(
                f"Epoch {epoch}/{self.vit_epochs}, Loss {self.vit_train_loss_per_epoch[epoch]}"
            )

            batch_pbar = tqdm(self.test_loader, leave=False)

            self.vit.eval()
            for batch in batch_pbar:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                cls_token = self.vit(x)
                loss = self.loss(cls_token, y)

                self.vit_test_loss_per_epoch[epoch] += loss.detach().cpu().item() / len(
                    self.test_loader
                )

        self.vit.eval()
        epoch_pbar = trange(self.head_epochs)

        for epoch in epoch_pbar:
            batch_pbar = tqdm(self.train_loader, leave=False)

            self.head.train()
            for batch in batch_pbar:
                self.head_opt.zero_grad()

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                cls_token = self.vit(x)
                pred = self.head(cls_token)
                loss = self.head_loss(pred, y)

                loss_float = loss.detach().cpu().item()
                self.head_train_loss_per_epoch[epoch] += loss_float / len(
                    self.train_loader
                )

                loss.backward()
                self.head_opt.step()

                batch_pbar.set_description(
                    f"Batch {b}/{len(self.train_loader)}, Loss {loss_float/self.batch_size}"
                )

            epoch_pbar.set_description(
                f"Epoch {epoch}/{self.head_epochs}, Loss {self.head_train_loss_per_epoch[epoch]}"
            )

            batch_pbar = tqdm(self.test_loader, leave=False)

            self.head.eval()
            for batch in batch_pbar:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                cls_token = self.vit(x)
                pred = self.head(cls_token)
                loss = self.head_loss(pred, y)

                self.head_test_loss_per_epoch[
                    epoch
                ] += loss.detach().cpu().item() / len(self.test_loader)

    def plot_training_stats(self):
        vit_epochs = np.arange(self.vit_epochs)
        plt.subplot(1, 2, 1)
        plt.title("ViT Training")
        plt.plot(vit_epochs, self.vit_train_loss_per_epoch)
        plt.plot(vit_epochs, self.vit_test_loss_per_epoch)
        plt.legend(["Training Loss", "Testing Loss"])

        head_epochs = np.arange(self.head_epochs)
        plt.subplot(1, 2, 1)
        plt.title("Classifier Head Training")
        plt.plot(head_epochs, self.head_train_loss_per_epoch)
        plt.plot(head_epochs, self.head_test_loss_per_epoch)
        plt.legend(["Training Loss", "Testing Loss"])

        plt.show()
