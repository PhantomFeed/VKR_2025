import torch
import torch.nn.utils
from torch.nn import CTCLoss
from tqdm import tqdm
from config import Config
from utils import TextEncoder, calculate_cer, calculate_wer
from synthtic import SyntheticTextGenerator

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = CTCLoss(blank=len(Config.charset)).to(device)
        self.text_encoder = TextEncoder(Config.charset)
        
        # Добавляем синтетический генератор
        # self.synth = synth
        # self.synthetic_generator = SyntheticTextGenerator(fonts_dir='data/fonts/')
        # self.synthetic_ratio = 0.2  # 20% синтетических данных в каждом батче
        
    def train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]', leave=True) as pbar:
            for images, texts in pbar:
                images = images.to(self.device)
                
                # batch_size = images.size(0)

                # num_synthetic = int(batch_size * self.synthetic_ratio)
                # if num_synthetic > 0:
                #     synthetic_images = []
                #     synthetic_texts = []
                    
                #     for _ in range(num_synthetic):
                #         img, text = self.synthetic_generator.generate_sample()
                #         img = self.train_loader.dataset.transform(img)  # применяем аугментации
                #         synthetic_images.append(img)
                #         synthetic_texts.append(text)
                    
                #     synthetic_images = torch.stack(synthetic_images).to(self.device)

                #     images = torch.cat([images, synthetic_images], dim=0)
                #     texts = tuple(list(texts) + synthetic_texts)
     
    
                targets, target_lengths = self.text_encoder.encode(texts)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
    
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                ).to(self.device)
    
                loss = self.criterion(
                    outputs.log_softmax(2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                
                loss.backward()

                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch_num):
        """Валидация с отображением прогресса"""
        self.model.eval()
        total_cer = 0
        total_wer = 0
        total_loss = 0
        total_samples = 0
        
        # Прогресс-бар для валидации
        with torch.no_grad():
            for images, texts in self.val_loader:
                images = images.to(self.device)
                
                targets, target_lengths = self.text_encoder.encode(texts)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Расчет CTC loss
                input_lengths = torch.full(
                    size=(outputs.size(1),),  # batch size
                    fill_value=outputs.size(0),  # sequence length
                    dtype=torch.int32
                ).to(self.device)
                
                loss = self.criterion(
                    outputs.log_softmax(2),  # [T, N, C]
                    targets,
                    input_lengths,
                    target_lengths
                )
                
                # Декодирование предсказаний
                _, pred_indices = torch.max(outputs.permute(1, 0, 2), 2)  # [B, W]
                pred_texts = [self.text_encoder.decode(indices) for indices in pred_indices]
                
                # Расчет метрик
                for pred, true in zip(pred_texts, texts):
                    total_cer += calculate_cer(pred, true)
                    total_wer += calculate_wer(pred, true)
                    total_samples += 1


                total_loss += loss.item()
                
                # Обновление прогресс-бара
                # pbar.set_postfix({
                #     'CER': f"{total_cer / total_samples:.4f}",
                #     'WER': f"{total_wer / total_samples:.4f}"
                # }, refresh=True)
                
        return total_loss / len(self.val_loader), total_cer / total_samples, total_wer / total_samples