import errno
import os
import stat
import subprocess
import shutil

import torch

from typing import Literal
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_image

def handleRemoveError(func, path, exc):
    # Windows causes some problems with file permissions which is why this error
    # handler is needed.
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

def intImgToViTFloat(img: torch.Tensor) -> torch.Tensor:
    return ((img.float() / 255.0) - 0.5) * 2.0

class WaldoDataset(Dataset):
    def __init__(self, img_res: Literal["64", "128", "256"], clear_cache=False, validation_split=0.1, training=True):
        assert img_res in ["64", "128", "256"], "img_res must be either 64, 128 or 256"
        url = "https://github.com/vc1492a/Hey-Waldo"
        
        if not Path("data", img_res).exists():
            subprocess.run(["git", "clone", url] )
            
            Path("data").mkdir(exist_ok=True)
            if clear_cache:
                shutil.rmtree(os.path.join("data", img_res), ignore_errors=True)

            shutil.copytree(os.path.join("Hey-Waldo", img_res), os.path.join("data", img_res))
            
            shutil.rmtree("Hey-Waldo",ignore_errors=False, onerror=handleRemoveError)
        
        self.data_path = os.path.join("data", img_res)
        self.img_names = []
        self.img_labels = []
        
        with_waldo = len(os.listdir(os.path.join(self.data_path, "waldo")))
        no_waldo = len(os.listdir(os.path.join(self.data_path, "notwaldo")))
        
        i = 0
        for p in os.listdir(os.path.join(self.data_path, "waldo")):
            if (training and i < (1 - validation_split) * with_waldo) or (not training and i >= (1 - validation_split) * with_waldo):
                self.img_names.append(os.path.join(self.data_path, "waldo", p))
                self.img_labels.append(1.0)
            i += 1

        j = 0
        for p in os.listdir(os.path.join(self.data_path, "notwaldo")):
            if (training and j < (1 - validation_split) * no_waldo) or (not training and j >= (1 - validation_split) * no_waldo):
                self.img_names.append(os.path.join(self.data_path, "notwaldo", p))
                self.img_labels.append(0.0)
            j += 1
        
        self.transform = intImgToViTFloat
        self.target_transform = None
        
        print(f"Waldo/Total = {with_waldo/(with_waldo+no_waldo)}")
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = self.img_names[index]
        image = read_image(img_path)
        label = self.img_labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label