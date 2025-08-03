from torch.utils.data import Dataset, ConcatDataset
import torch
from transformers import CanineTokenizer, CanineModel
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
import re
import json
import pandas as pd
import torchvision.transforms as transforms
from modules.utils.config import Config
import os
import torch
import gc
import numpy as np
import bisect
import h5py
import string

# SAVE_PATH = "/home/fantom/Диплом/DiffusionPen/processed_data.pt"

# # Функции сохранения и загрузки предобработанных данных
# def save_data(data, path=SAVE_PATH):
#     torch.save(data, path)
#     print(f"Preprocessed data saved to {path}")

# def load_data(path=SAVE_PATH):
#     if os.path.exists(path):
#         print(f"Loading cached data from {path}...")
#         return torch.load(path)
#     return None

# Класс для загрузки данных из Cyrillic датасета
# class CyrillicDataset(Dataset):
#     def __init__(self, root, tsv_path, transform=None):
#         self.root = root
#         self.data = pd.read_csv(tsv_path, sep='\t', header=0).dropna()
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_name, text = self.data.iloc[idx]
#         img_path = os.path.join(self.root, img_name)
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, text, img_path

# # Класс для загрузки данных из HKR датасета
# class HKRDataset(Dataset):
#     def __init__(self, img_dir, ann_dir, transform=None):
#         self.img_dir = img_dir
#         self.ann_dir = ann_dir
#         self.transform = transform
#         self.annotations = self._load_annotations()
    
#     def checkkz(self, text):
#         return all(char not in Config.kzcharset for char in text)
    
#     def _load_annotations(self):
#         anns = []
#         for fname in os.listdir(self.ann_dir):
#             with open(os.path.join(self.ann_dir, fname)) as f:
#                 ann = json.load(f)
#                 if self.checkkz(ann['description']):
#                     anns.append((ann['name'], ann['description']))
#         return anns
    
#     def __len__(self):
#         return len(self.annotations)
    
#     def __getitem__(self, idx):
#         img_name, text = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, text, img_path


class CyrillicDataset(Dataset):
    def __init__(self, tsv_path):
        self.data = pd.read_csv(tsv_path, sep='\t', header=0).dropna()
        # self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, text = self.data.iloc[idx]
        # img_path = os.path.join(self.root, img_name)
        # image = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        return text

# Класс для загрузки данных из HKR датасета
class HKRDataset(Dataset):
    def __init__(self, ann_dir):
        self.ann_dir = ann_dir
        self.annotations = self._load_annotations()
    
    def checkkz(self, text):
        return all(char not in Config.kzcharset for char in text)
    
    def _load_annotations(self):
        anns = []
        for fname in os.listdir(self.ann_dir):
            with open(os.path.join(self.ann_dir, fname)) as f:
                ann = json.load(f)
                if self.checkkz(ann['description']):
                    anns.append((ann['description']))
        return anns
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        text = self.annotations[idx]
        return text


# class CachedLatentDataset(torch.utils.data.Dataset):
#     def __init__(self, cached_dir):
#         self.cached_dir = cached_dir
#         self.file_list = sorted([f for f in os.listdir(cached_dir) if f.endswith('.pt')])

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         data = torch.load(os.path.join(self.cached_dir, self.file_list[idx]))
#         return data["latent"], data["text_emb"], data["path"]

class HDF5LazyDataset(Dataset):
    def __init__(self, path, tsv_path, ann_dir):
        self.path = path
        self.h5 = None
        self.cyrillic_dataset = CyrillicDataset(tsv_path)
        self.hkr_dataset = HKRDataset(ann_dir)

        # self.data_sources = [("cyrillic", i) for i in range(len(self.cyrillic_dataset))] + \
        #                     [("hkr", i) for i in range(len(self.hkr_dataset))]
        transcrs = ''.join([self.cyrillic_dataset[i] for i in range(len(self.cyrillic_dataset))] +
                           [self.hkr_dataset[i] for i in range(len(self.hkr_dataset))])
        self.alphabet = sorted(list(set(transcrs)))

    def _init_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")
            self.latents = self.h5["latents"]
            self.text_embs = self.h5["text_embs"]
            self.paths = self.h5["paths"]

    def __len__(self):
        self._init_h5()
        return len(self.latents)

    def __getitem__(self, idx):
        self._init_h5()
        return {
            "latent": torch.tensor(self.latents[idx]),
            "text_emb": torch.tensor(self.text_embs[idx]),
            "path": self.paths[idx].decode("utf-8") if isinstance(self.paths[idx], bytes) else self.paths[idx]
        }



# class CustomDataset(Dataset):
#     def __init__(self, hkr_img_dir, hkr_ann_dir, cyrillic_train_root, cyrillic_train_tsv_path, transforms=None, 
#                  stable_dif_path='stable-diffusion-v1-5/stable-diffusion-v1-5', device='cuda:0'):
#         super(CustomDataset, self).__init__()
#         self.transforms = transforms
#         self.device = device

#         # Tokenizer and encoder
#         self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
#         self.text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
#         self.text_encoder.eval()

#         # VAE
#         self.vae = AutoencoderKL.from_pretrained(stable_dif_path, subfolder="vae").to(device)
#         self.vae.eval()
#         self.vae.requires_grad_(False)

#         # Загружаем датасеты
#         self.cyrillic_dataset = CyrillicDataset(cyrillic_train_root, cyrillic_train_tsv_path)
#         self.hkr_dataset = HKRDataset(hkr_img_dir, hkr_ann_dir)

#         self.data_sources = [("cyrillic", i) for i in range(len(self.cyrillic_dataset))] + \
#                             [("hkr", i) for i in range(len(self.hkr_dataset))]

#         # Собираем алфавит только один раз
#         transcrs = ''.join([self.cyrillic_dataset[i][1] for i in range(len(self.cyrillic_dataset))] +
#                            [self.hkr_dataset[i][1] for i in range(len(self.hkr_dataset))])
#         self.alphabet = sorted(list(set(transcrs)))

#     def __len__(self):
#         return len(self.data_sources)

#     def __getitem__(self, index):
#         source, real_idx = self.data_sources[index]
#         if source == "cyrillic":
#             image, text, img_path = self.cyrillic_dataset[real_idx]
#         else:
#             image, text, img_path = self.hkr_dataset[real_idx]

#         if self.transforms:
#             image = self.transforms(image)

#         # Преобразование текста
#         # with torch.no_grad():
#         tokenized = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(self.device)
#         text_features = self.text_encoder(**tokenized).last_hidden_state.squeeze(0).cpu()

#         # Преобразование изображения
#         # with torch.no_grad():
#         image_tensor = image.to(self.device).unsqueeze(0).float()  # Add batch dim
#         latent = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
#         latent = latent.squeeze(0).cpu()

#         return latent, text_features, img_path

# from torch.utils.data import Dataset
# import torch
# from transformers import CanineTokenizer, CanineModel
# from diffusers import AutoencoderKL
# from PIL import Image
# import os
# import pandas as pd
# from tqdm import tqdm

# # Класс для загрузки данных из Cyrillic датасета
# class CyrillicDataset(Dataset):
#     def __init__(self, root, tsv_path, transform=None):
#         self.root = root
#         self.data = pd.read_csv(tsv_path, sep='\t', header=0).dropna()
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_name, text = self.data.iloc[idx]
#         img_path = os.path.join(self.root, img_name)
#         image = Image.open(img_path).convert('RGB')
#         # if self.transform:
#         #     image = self.transform(image)
#         return image, text, img_path

# # Класс для загрузки данных из HKR датасета
# class HKRDataset(Dataset):
#     def __init__(self, img_dir, ann_dir, transform=None):
#         self.img_dir = img_dir
#         self.ann_dir = ann_dir
#         self.transform = transform
#         self.annotations = self._load_annotations()
    
#     def checkkz(self, text):
#         return all(char not in Config.kzcharset for char in text)
    
#     def _load_annotations(self):
#         anns = []
#         for fname in os.listdir(self.ann_dir):
#             with open(os.path.join(self.ann_dir, fname)) as f:
#                 ann = json.load(f)
#                 if self.checkkz(ann['description']):
#                     anns.append((ann['name'], ann['description']))
#         return anns
    
#     def __len__(self):
#         return len(self.annotations)
    
#     def __getitem__(self, idx):
#         img_name, text = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
#         image = Image.open(img_path).convert('RGB')
#         # if self.transform:
#         #     image = self.transform(image)
#         return image, text, img_path

# class CustomDataset(Dataset):
#     def __init__(self, hkr_img_dir, hkr_ann_dir, cyrillic_train_root, cyrillic_train_tsv_path, transforms=None, 
#                  stable_dif_path='stable-diffusion-v1-5/stable-diffusion-v1-5', device='cuda:0'):
#         super().__init__()
#         self.hkr_img_dir = hkr_img_dir
#         self.hkr_ann_dir = hkr_ann_dir
#         self.cyrillic_train_root = cyrillic_train_root
#         self.cyrillic_train_tsv_path = cyrillic_train_tsv_path
#         self.transforms = transforms
#         self.device = device

#         self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
#         self.text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
#         self.vae = AutoencoderKL.from_pretrained(stable_dif_path, subfolder="vae").to(device)
#         self.vae.requires_grad_(False)

#         self.cyrillic_dataset = CyrillicDataset(cyrillic_train_root, cyrillic_train_tsv_path, transforms)
#         self.hkr_dataset = HKRDataset(hkr_img_dir, hkr_ann_dir, transforms)
#         self.data = torch.utils.data.ConcatDataset([self.cyrillic_dataset, self.hkr_dataset])
        
#         self.alphabet = self._build_alphabet()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image, text, img_path = self.data[idx]

#         # Преобразование изображения
#         if self.transforms:
#             image = self.transforms(image)

#         # Перенос всего процесса VAE и токенизации сюда
#         with torch.no_grad():
#             text_features = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(self.device)
#             text_features = self.text_encoder(**text_features).last_hidden_state.cpu()
            
#             image = image.unsqueeze(0).to(self.device)
#             image = self.vae.encode(image.to(torch.float32)).latent_dist.sample()
#             image = image * 0.18215
#             image = image.cpu().squeeze(0)

#         return image, text_features, img_path
        
#     def _build_alphabet(self):
#         print("Building alphabet...")
#         transcrs = []
#         for i in tqdm(range(len(self.data))):
#             _, text, _ = self.data[i]
#             transcrs.append(text)
#         transcrs = ''.join(transcrs)
#         alphabet = sorted(list(set(transcrs)))
#         print("Alphabet built.")
#         return alphabet
