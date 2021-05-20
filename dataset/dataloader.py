import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob
from typing import Union
from PIL import Image
import numpy as np

class LoadDataset(Dataset):
    def __init__(self, data_path, transform, mode='train'):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        if mode == "test":
            self.test_load()
        else : 
            self.load_dataset()

    def test_load(self):
        root = os.path.join(self.data_path, self.mode)
        print("root : ", root)
        self.data = glob.glob(root+"/*.png")
        
    def load_dataset(self):
        root = os.path.join(self.data_path, self.mode)
        print("root : ", root)
        self.data = ImageFolder(root=root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == "test":
            img = Image.open(self.data[index]).convert('RGB')
            img = self.transform(img)
            return img
        else:
            img, label = self.data[index]
            img, label = self.transform(img), int(label)
            return img, label


# --- UTILITY FUNCTIONS ---
def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

class IMDataset(Dataset):
    def __init__(self, data_path, transform):
        super(IMDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path)
        print("root : ", root)
        path_list = glob.glob(root + "/*/*.png")
        self.len = len(path_list)

        path_tmp = [string_to_sequence(s) for s in path_list]
        self.path_v, self.path_o = pack_sequences(path_tmp)


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        seq = unpack_sequence(self.path_v, self.path_o, index)
        data_path = sequence_to_string(seq)
        
        img = Image.open(data_path).convert('RGB')
        img = self.transform(img)
        return img