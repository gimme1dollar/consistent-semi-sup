import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob
from PIL import Image
from typing import Union
import numpy as np

def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]):
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

def path_join(train_path, label, file_list):
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(train_path, label, f))
    
    return path_list
class LoadDataset(Dataset):
    def __init__(self, data_path, transform, mode='valid'):
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
        train_path = os.path.join(self.data_path, self.mode)
        folder_list = os.listdir(train_path) # folder list [0,1,2,...,19]

        path_list = []
        for label_num in folder_list:
            file_path = os.path.join(train_path, label_num)     
            file_list = os.listdir(file_path)
            path_list += path_join(train_path, label_num, file_list)
        self.image_len = len(path_list)
        img_seq = [string_to_sequence(s) for s in path_list]
        self.image_v, self.image_o = pack_sequences(img_seq)
        img_seq

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        if self.mode == "test":
            img = Image.open(self.data[index]).convert('RGB')
            img = self.transform(img)
            return img
        else:
            path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
            label = int(path.split("/")[-2])
            img = Image.open(path).convert('RGB')
            img = self.transform(img)

            return img, label

class LoadSemiDataset(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(LoadSemiDataset, self).__init__()
        self.data_path = data_path
        self.list_name = str(ratio)+"_"+mode+"_path_list.txt"
        self.mode = mode
        self.transform = transform
        
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path, self.list_name)
        print(root)
        with open(os.path.join(self.data_path, self.list_name), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.image_len = len(file_names)
        img_seq = [string_to_sequence(s) for s in file_names]
        self.image_v, self.image_o = pack_sequences(img_seq)

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
        label = int(path.split("/")[-2])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if self.mode == 'label':
            return img, label
        elif self.mode == 'unlabel':
            return img, label, path
        else: 
            raise NotImplementedError()

class LoadSemiDataset2(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(LoadSemiDataset2, self).__init__()
        self.data_path = data_path
        self.list_name = str(ratio)+"_"+mode+"_path_list.txt"
        if mode == 'unlabel':
            self.list_name = str(ratio)+"_"+mode+"_path_list2.txt"
            self.files={}
            self.files_len = []
            self.label = []
        self.mode = mode
        self.transform = transform
        
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path, self.list_name)
        print(root)
        if self.mode == 'label':
            with open(os.path.join(self.data_path, self.list_name), "r") as f:
                file_names = [x.strip() for x in f.readlines()] 
            self.image_len = len(file_names)
            img_seq = [string_to_sequence(s) for s in file_names]
            self.image_v, self.image_o = pack_sequences(img_seq)
        else:
            with open(os.path.join(self.data_path, self.list_name), "r") as f:
                files = {}
                files_len= []
                for k in range(20):
                    files[(k)] = []
                img_num = 0
                for x in f.readlines():
                    img_num+=1
                    words = (x.strip()).split(' ')
                    p = words[0]
                    l = int(words[1])
                    files[l].append(p)
                self.files = files
                for i in range(20):
                    num = len(files[i])
                    files_len.append(num)
                self.files_len = files_len
                self.image_len = img_num
        

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        if self.mode == 'label':
            path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
            label = int(path.split("/")[-2])
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label
        elif self.mode == 'unlabel':
            l = self.label[0]
            # self.label = np.delete(self.label, 0)
            idx = np.random.randint(0, self.files_len[l])
            path = self.files[l][idx]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, l
        else: 
            raise NotImplementedError()

class LoadSemiDataset3(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(LoadSemiDataset3, self).__init__()
        self.data_path = data_path
        self.list_name1 = str(ratio)+"_"+'label'+"_path_list.txt"
        self.list_name2 = str(ratio)+"_"+'unlabel'+"_path_list2.txt"
        self.files={}
        self.files_len = []
        self.mode = mode
        self.transform = transform
        
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path, self.list_name1)
        print(root)
        with open(os.path.join(self.data_path, self.list_name1), "r") as f:
            file_names = [x.strip() for x in f.readlines()] 
        self.image_len = len(file_names)
        img_seq = [string_to_sequence(s) for s in file_names]
        self.image_v, self.image_o = pack_sequences(img_seq)

        with open(os.path.join(self.data_path, self.list_name2), "r") as f:
            files = {}
            files_len= []
            for k in range(20):
                files[(k)] = []
            img_num = 0
            for x in f.readlines():
                img_num+=1
                words = (x.strip()).split(' ')
                p = words[0]
                l = int(words[1])
                files[l].append(p)
            self.files = files
            for i in range(20):
                num = len(files[i])
                files_len.append(num)
            self.files_len = files_len
        

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        
        path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
        label = int(path.split("/")[-2])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        idx = np.random.randint(0, self.files_len[label])
        path2 = self.files[label][idx]
        img2 = Image.open(path2).convert('RGB')
        img2 = self.transform(img2)
        return img, img2, label


class LoadSemiDataset4(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(LoadSemiDataset3, self).__init__()
        self.data_path = data_path
        
        self.list_name1 = str(ratio)+"_"+'label'+"_path_list.txt"
        self.list_name2 = str(ratio)+"_"+'unlabel'+"_path_list2.txt"
        self.files={}
        self.files_len = []
        self.mode = mode
        self.transform = transform
        
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path, self.list_name1)
        print(root)
        with open(os.path.join(self.data_path, self.list_name1), "r") as f:
            file_names = [x.strip() for x in f.readlines()] 
        self.image_len = len(file_names)
        img_seq = [string_to_sequence(s) for s in file_names]
        self.image_v, self.image_o = pack_sequences(img_seq)

        with open(os.path.join(self.data_path, self.list_name2), "r") as f:
            files = {}
            files_len= []
            for k in range(20):
                files[(k)] = []
            img_num = 0
            for x in f.readlines():
                img_num+=1
                words = (x.strip()).split(' ')
                p = words[0]
                l = int(words[1])
                files[l].append(p)
            self.files = files
            for i in range(20):
                num = len(files[i])
                files_len.append(num)
            self.files_len = files_len
        

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        
        path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
        label = int(path.split("/")[-2])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        idx = np.random.randint(0, self.files_len[label])
        path2 = self.files[label][idx]
        img2 = Image.open(path2).convert('RGB')
        img2 = self.transform(img2)
        return img, img2, label

