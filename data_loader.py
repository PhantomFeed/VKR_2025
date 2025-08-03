import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from config import Config

class CyrillicDataset(Dataset):
    def __init__(self, root, tsv_path, transform=None):
        self.root = root
        self.data = pd.read_csv(tsv_path, sep='\t', header=0)
        self.data = self.data.dropna()  # Удаляем строки с NaN
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, text = self.data.iloc[idx]
        text = text.lower()

        if pd.isna(text) or not isinstance(text, str):
            print(f"Warning: Invalid text in Cyrillic at index {idx}: {text}")
            text = ""  # Заменяем на пустую строку
        
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, text

class MineDataset(Dataset):
    def __init__(self, root, tsv_path, transform=None):
        self.root = root
        self.data = pd.read_csv(tsv_path, sep='\t', header=0)
        self.data = self.data.dropna()  # Удаляем строки с NaN
        # self.data = self.data[self.data[1].apply(lambda x: isinstance(x, str))] # Оставляем только строки
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, text = self.data.iloc[idx]
        text = text.lower()

        if pd.isna(text) or not isinstance(text, str):
            print(f"Warning: Invalid text in Cyrillic at index {idx}: {text}")
            text = ""  # Заменяем на пустую строку
        
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, text

class HKRDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.annotations = self._load_annotations()

    def checkkz(self, text):
        for char in text:
            if char in Config.kzcharset:
                return False
        return True
    
    def _load_annotations(self):
        anns = []
        for fname in os.listdir(self.ann_dir):
            with open(os.path.join(self.ann_dir, fname)) as f:
                ann = json.load(f)
                if self.checkkz(ann['description']):
                    anns.append((ann['name'], ann['description']))
        return anns
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name, text = self.annotations[idx]
        text = text.lower()
        
        # Проверка на NaN и тип данных
        
        if pd.isna(text) or not isinstance(text, str):
            print(f"Warning: Invalid text in HKR at index {idx}: {text}")
            text = ""  # Заменяем на пустую строку
        
        img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, text

def get_augmented_transforms(input_size):
    return transforms.Compose([
        transforms.AugMix(severity=2),
        transforms.Grayscale(),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_transforms(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])