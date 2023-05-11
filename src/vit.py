import numpy as np
import torch
import torch.nn as nn

from waldo import WaldoDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

def patchify(images, n_patches):
    # Batch, Color Channels, Height, Width
    n, c, h, w = images.shape

    assert h == w

    # Batch, Number of Images, Size of each Patch
    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j+1)*patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i][j] = np.cos(i / (10000 ** (j / d)))

    return result

class ViT(nn.Module):
    def __init__(self, colors, height, width, n_patches, hidden_dimension, n_heads, n_blocks, out_d):
        super(ViT, self).__init__()

        self.c = colors
        self.h = height
        self.w = width

        self.n_patches = n_patches
        self.hidden_dimension = hidden_dimension

        assert self.h % self.n_patches == 0
        assert self.w % self.n_patches == 0
        self.patch_size = (self.h // n_patches, self.w // n_patches)

        # Linear mapper
        self.input_dimension = self.c * self.patch_size[0] * self.patch_size[1]
        self.linear1 = nn.Linear(self.input_dimension, self.hidden_dimension)

        # Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dimension))

        # Positional embedding
        self.register_buffer("pos_embed", get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dimension), persistent=False)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(self.hidden_dimension, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dimension, out_d),
            # nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)
        tokens = self.linear1(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        out = tokens + self.pos_embed.repeat(n, 1, 1)
        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        return self.mlp(out)

class ViTNoHead(nn.Module):
    def __init__(self, colors, height, width, n_patches, hidden_dimension, n_heads, n_blocks):
        super(ViTNoHead, self).__init__()

        self.c = colors
        self.h = height
        self.w = width

        self.n_patches = n_patches
        self.hidden_dimension = hidden_dimension

        assert self.h % self.n_patches == 0
        assert self.w % self.n_patches == 0
        self.patch_size = (self.h // n_patches, self.w // n_patches)

        # Linear mapper
        self.input_dimension = self.c * self.patch_size[0] * self.patch_size[1]
        self.linear1 = nn.Linear(self.input_dimension, self.hidden_dimension)

        # Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dimension))

        # Positional embedding
        self.register_buffer("pos_embed", get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dimension), persistent=False)

        # Transformer encoder blocks

        self.blocks = nn.ModuleList([ViTBlock(self.hidden_dimension, n_heads) for _ in range(n_blocks)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)
        tokens = self.linear1(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        out = tokens + self.pos_embed.repeat(n, 1, 1)
        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        # return self.softmax(out)
        return out

    def attention_rollout(self, image):
        # https://storrs.io/attention-rollout/
        self.eval()
        n, c, h, w = image.shape
        assert n == 1

        patches = patchify(image, self.n_patches).to(self.pos_embed.device)
        tokens = self.linear1(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        out = tokens + self.pos_embed.repeat(n, 1, 1)
        for b, block in enumerate(self.blocks):
            if b == 0:
                A = block.attention(out, self.n_patches, self.n_patches)
            else:
                A = block.attention(out, self.n_patches, self.n_patches) @ A
            out = block(out)

        return A


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0

        d_head = d // n_heads
        self.d_head = d_head
        self.q = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q[head]
                k_mapping = self.k[head]
                v_mapping = self.v[head]

                seq = sequence[:, head * self.d_head: (head+1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

    def attention(self, sequence, w: int, h: int) -> np.ndarray:
        # THIS FUNCTION ONLY TAKES ONE SEQUENCE AT A TIME
        assert sequence.shape[0] == 1

        sequence = sequence[0]
        attentions = []
        for head in range(self.n_heads):
            q_mapping = self.q[head]
            k_mapping = self.k[head]

            seq = sequence[:, head * self.d_head: (head+1) * self.d_head]
            q, k = q_mapping(seq), k_mapping(seq)

            attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
            attentions.append(attention[0,:])

        # TODO (Vincent): Check if the attentions are actually stored in this
        # order. The patches are probably stored row by row in succession. That
        # should entail a (height, width) ordering of the output.
        att_map = np.zeros(w * h)
        for i, att_row in enumerate(attentions):
            att_map[i*(w//self.n_heads)**2:(i+1)*(w//self.n_heads)**2] = att_row[1:].detach()

        return np.reshape(att_map, (h, w))

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

    def attention(self, x, w, h):
        return self.mhsa.attention(x, w, h)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViT(3, 64, 64, 16, 32, 4, 3, 1).to(device)

    train_data = WaldoDataset("64", validation_split=0.1, training=True)
    test_data = WaldoDataset("64", validation_split=0.1, training=False)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=64)

    print(len(train_data), len(test_data))

    EPOCHS = 15
    LR = 0.005

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    for epoch in trange(EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), torch.unsqueeze(y.to(device), -1).to(torch.float32)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} loss: {train_loss:.12f}")

    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), torch.unsqueeze(y.to(device), -1).to(torch.float32)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        test_loss += loss.detach().cpu().item() / len(train_loader)
        total += len(x)
        correct += (torch.round(y_hat) == y).detach().cpu().sum()

    print(f"Test loss: {test_loss}")
    print(f"Test Accuracy: {correct / total:.12f}  {correct} {total}")

    for batch in train_loader:
        model.eval()
        x, y = batch
        x, y = x.to(device), torch.unsqueeze(y.to(device), -1)
        y_hat = model(x)
        print(x.shape, y.shape, y_hat.shape)

        print(y_hat[0,:], y[0,:])
        plt.imshow((x[0,:,:,:].permute(1, 2, 0) + 1.0)/2.0)
        plt.show()
        break
