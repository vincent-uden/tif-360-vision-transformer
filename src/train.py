import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from vit import ViTNoHead
from gen_data import GeneratedDataset
from tqdm import tqdm, trange
from sklearn.manifold import TSNE

# [/] Create ViT With Separate Classifier
# [ ] Implement a trainer class for unsupervised learning
#     [x] Data Loading
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
        vit_lr=0.005,
        head_lr=0.005,
        vit_epochs=5,
        head_epochs=5,
        batch_size=4,
    ):
        self.vit = vit
        self.head = head

        self.vit_opt = torch.optim.SGD(self.vit.parameters(), lr=vit_lr, momentum=0.1)
        self.head_opt = torch.optim.SGD(
            self.head.parameters(), lr=head_lr, momentum=0.1
        )

        self.head_loss = torch.nn.BCELoss()

        self.batch_size = batch_size
        self.vit_epochs = vit_epochs
        self.head_epochs = head_epochs
        self.device = device
        assert self.batch_size % 4 == 0, "Batch size must be divisible by 4"

        self.train_data = GeneratedDataset(256)
        self.test_data = GeneratedDataset(128, start_i=256)

        self.train_loader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=True
        )

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

        # print("-------------------------")
        # print(cls_token)
        # print(label)

        margin = 5
        euclid_distances = torch.cdist(cls_token, cls_token)
        # print("OUTPUT")
        # print(output_similarities)

        # 0 implies similar
        target_similarities = 1 - torch.mm(label, torch.transpose(label, 0, 1))

        # print("TARGET")
        # # print(target_similarities)
        # print(target_unsimilarities)

        # combined_scores = torch.triu(combined_scores, diagonal=1)
        # print("SCORES")
        # print(combined_scores)

        loss = (
            1 - target_similarities
        ) * 0.5 * euclid_distances**2 + target_similarities * 0.5 * torch.maximum(
            margin - euclid_distances,
            torch.tensor(0).expand_as(euclid_distances).to(self.device),
        ) ** 2

        return torch.sum(loss)

    def accuracy(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # I expect the label to be on the form
        #   [[1, 0]]: Waldo
        #   [[0, 1]]: Not Waldo
        # with dimension (batch_size, 2)

        cls_token = self.vit(data)
        pred = self.head(cls_token)

        correct_by_category = torch.sum(pred == labels, dim=0)

        return correct_by_category

    def train_vit(self):
        print("Training Vision Transformer")
        epoch_pbar = trange(self.vit_epochs)

        for epoch in epoch_pbar:
            batch_pbar = tqdm(self.train_loader, leave=False)

            self.vit.train()
            for batch in batch_pbar:
                self.vit_opt.zero_grad()

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                cls_token = self.vit(x)
                loss = self.loss(cls_token, y)

                loss_float = loss.detach().cpu().item()
                self.vit_train_loss_per_epoch[epoch] += loss_float / (
                    len(self.train_loader) * self.batch_size
                )

                loss.backward()
                self.vit_opt.step()

                batch_pbar.set_description(f"Loss {loss_float/self.batch_size:.10f}")

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

                self.vit_test_loss_per_epoch[epoch] += loss.detach().cpu().item() / (
                    len(self.test_loader) * self.batch_size
                )

    def cache_vit_outputs(self, dataloader: DataLoader) -> DataLoader:
        head_X = torch.zeros((len(dataloader) * self.batch_size, self.vit.hidden_dimension)).cpu()
        head_Y = torch.zeros((len(dataloader) * self.batch_size, 2)).cpu()

        b = 0
        for batch in tqdm(dataloader, position=0, leave=True):
            x, y = batch
            x = x.to(self.device)

            head_X[b*self.batch_size:(b+1)*self.batch_size,:] = self.vit(x).detach().cpu()
            head_Y[b*self.batch_size:(b+1)*self.batch_size,:] = y
            b += 1

            del x

        dset = TensorDataset(head_X, head_Y)
        return DataLoader(dset, batch_size=self.batch_size, shuffle=True)

    def train_head(self):
        self.vit.eval()

        print("Caching training data for head training")
        head_train = self.cache_vit_outputs(self.train_loader)
        print("Caching testing data for head training")
        head_test = self.cache_vit_outputs(self.test_loader)
        print("Training Classifier Head")

        epoch_pbar = trange(self.head_epochs)
        self.head.train()
        for epoch in epoch_pbar:
            batch_pbar = tqdm(head_train, leave=False)

            for batch in batch_pbar:
                self.head_opt.zero_grad()

                x, y = batch
                cls_token, y = x.to(self.device), y.to(self.device)

                pred = self.head(cls_token)
                loss = self.head_loss(pred, y)

                loss_float = loss.detach().cpu().item()
                self.head_train_loss_per_epoch[epoch] += loss_float / len(
                    self.train_loader
                )

                loss.backward()
                self.head_opt.step()

                batch_pbar.set_description(
                    f"Loss {loss_float/self.batch_size}"
                )

            epoch_pbar.set_description(
                f"Epoch {epoch}/{self.head_epochs}, Loss {self.head_train_loss_per_epoch[epoch]}"
            )

            batch_pbar = tqdm(head_test, leave=False)

            self.head.eval()
            for batch in batch_pbar:
                x, y = batch
                cls_token, y = x.to(self.device), y.to(self.device)

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
        plt.subplot(1, 2, 2)
        plt.title("Classifier Head Training")
        plt.plot(head_epochs, self.head_train_loss_per_epoch)
        plt.plot(head_epochs, self.head_test_loss_per_epoch)
        plt.legend(["Training Loss", "Testing Loss"])

        plt.show()

    def tsne_vit_plot(self):
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        self.vit.eval()

        tokens = []
        colors = []
        labels = []
        values = []
        for batch in tqdm(self.test_loader):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            for row in y:
                if row[0]:
                    colors.append((1, 0, 0))
                    labels.append("Waldo")
                    values.append(1)
                else:
                    colors.append((0, 0, 1))
                    labels.append("Not Waldo")
                    values.append(0)
            cls_token = self.vit(x).detach().cpu().numpy()
            tokens.append(cls_token)

        tokens = np.concatenate(tokens)
        z = tsne.fit_transform(tokens)

        scatter = plt.scatter(z[:,0], z[:,1], c=values, cmap="rainbow")
        plt.legend(handles=scatter.legend_elements()[0], labels=["Waldo", "Not Waldo"])
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_h = 32
    model = ViTNoHead(
        colors=3,
        height=256,
        width=256,
        n_patches=16,
        hidden_dimension=d_h,
        n_heads=8,
        n_blocks=10,
    )
    head = torch.nn.Sequential(
        torch.nn.Linear(d_h, 2),
        torch.nn.Sigmoid(),
    )

    model.load_state_dict(torch.load("vitnohead_shuffle2.pt"))

    model.to(device)
    head.to(device)

    trainer = UnsupervisedTrainer(
        model,
        head,
        device,
        vit_epochs=300,
        batch_size=4,
        vit_lr=0.001,
        head_lr=0.001,
        head_epochs=50,
    )

    trainer.tsne_vit_plot()

    # trainer.train_vit()
    # torch.save(model.state_dict(), "vitnohead_shuffle2.pt")
    trainer.train_head()
    torch.save(head.state_dict(), "head_shuffle3.pt")
    trainer.plot_training_stats()

