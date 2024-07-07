import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from PIL import Image


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(len(self.X))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
   
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annotations_file, img_dir:str = "images", transform=None, target_transform=None) -> None:
        self.img_paths = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.X = torch.load(os.path.join(data_path))
        with open(annotations_file, "r") as file:
            for line in file:
                self.img_paths.append(line.strip())
            print(len(self.img_paths))

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        eeg_data = self.X[idx]
        img_path = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # 画像に変換を適用
        return eeg_data, image

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


        