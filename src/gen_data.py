import torch
import torchvision
import os
from torchvision import transforms
from os import path, mkdir
from PIL import Image, ImageOps
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

convert_tensor = transforms.ToTensor()
images = 16 #Amount of images you want to generate. Can be modified
background = 125
char  = 3
count = 0
min_char_size = 50
max_char_size = 150

class GeneratedDataset(Dataset):
    def __init__(self, images, start_i = 0):
        self.images = images
        self.start_i = start_i

    def __len__(self):
        return self.images

    def __getitem__(self, index):
        # Deterministic on-demand image generation
        assert index < len(self)
        index += self.start_i

        np.random.seed(index)
        random.seed(index)

        bg_rand = np.random.randint(1, background + 1)
        bgString = '../data/background/nature ({}) (Copy).jpg'.format(bg_rand)
        char_rand = np.random.randint(0, char)
        size_rand = np.random.randint(min_char_size, max_char_size)

        if index % 2 == 0:
            charString = '../data/waldo/char{}.png'.format(char_rand)
            label = torch.tensor([1, 0], dtype=torch.int32)
        else:
            charString = '../data/cat/notchar{}.png'.format(char_rand)
            label = torch.tensor([0, 1], dtype=torch.int32)

        bg = Image.open(bgString).convert("RGB")
        bg_x, bg_y = bg.size
        im = Image.open(charString)
        im = im.resize((size_rand,size_rand))
        # rand_pert = np.random.randint(0, 4)
        rand_pert = 3

        if rand_pert == 0:
            im = ImageOps.flip(im)
        elif rand_pert == 1:
            im = ImageOps.mirror(im)
        elif rand_pert == 2:
            im = ImageOps.flip(im)
            im = ImageOps.mirror(im)

        im_x, im_y = im.size
        rand_x = random.randint(0, bg_x-im_x)
        rand_y = random.randint(0, bg_y-im_y)
        bg.paste(im,(rand_x,rand_y),mask=im)
        bg_tensor = convert_tensor(bg)

        image, label = bg_tensor, label

        return image.to(torch.float32), label.to(torch.float32)
