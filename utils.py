import torch
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import distance
import matplotlib.pyplot as plt
from config import *

class TextEncoder:
    """Кодирование текста для CTC loss"""
    def __init__(self, charset):
        self.charset = charset
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}
        self.blank_idx = len(charset)  # Индекс для blank символа в CTC

    def encode(self, texts):
        """Преобразование текста в тензоры"""
        encoded_texts = []
        lengths = []
        
        for text in texts:
            # Фильтрация символов, не входящих в charset
            encoded = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
            encoded_texts.append(torch.IntTensor(encoded))
            lengths.append(len(encoded))
        
        # Паддинг до максимальной длины в батче
        padded = pad_sequence(encoded_texts, batch_first=True)
        return padded, torch.IntTensor(lengths)

    def decode(self, indices):
        """Декодирование индексов в текст"""
        return ''.join([self.charset[idx] for idx in indices if idx != self.blank_idx])

def calculate_cer(pred, target):
    """Character Error Rate"""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return distance(pred, target) / len(target)

def calculate_wer(pred, target):
    """Word Error Rate"""
    pred_words = pred.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return distance(pred_words, target_words) / len(target_words)

# def calculate_cer(pred, true):
#     return sum(p != t for p, t in zip(pred, true)) / max(len(true), 1)

# def calculate_wer(pred, true):
#     pred_words = pred.split()
#     true_words = true.split()
#     return sum(p != t for p, t in zip(pred_words, true_words)) / max(len(true_words), 1)

def plot_metrics(train_losses, val_losses, val_cer, val_wer):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_cer, label='CER')
    plt.plot(val_wer, label='WER')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.show()

def visualize_predictions(model, val_loader, text_encoder, num_samples=5):
    model.eval()
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(config.device)
            outputs = model(images)
            _, pred_indices = torch.max(outputs.permute(1, 0, 2), 2)
            pred_texts = [text_encoder.decode(indices) for indices in pred_indices]
            
            for i in range(min(num_samples, len(texts))):
                print(f"True: {texts[i]}")
                print(f"Pred: {pred_texts[i]}")
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.show()
            break