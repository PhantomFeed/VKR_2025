import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, text = self.dataset[idx]
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
# --------------------------------
# class RandomAugmentation:
#     def __init__(self):
#         self.augmentations = [
#             transforms.RandomRotation(degrees=(-10, 10)),
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
#         ]

#     def __call__(self, img):
#         num_augmentations = random.randint(1, 3)  # Выбираем от 1 до 3 аугментаций
#         selected_augmentations = random.sample(self.augmentations, num_augmentations)
#         transform = transforms.Compose(selected_augmentations)
#         return transform(img)

# def get_augmented_transforms(input_size):
#     return transforms.Compose([
#         RandomAugmentation(),
#         transforms.Resize(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])