import os
import string
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
from diffusers import AutoencoderKL, DDIMScheduler
from modules.models.unet import UNetModel
from torchvision import transforms
from modules.utils.custom_dataset import *
from modules.utils.iam_dataset import IAMDataset
from modules.utils.GNHK_dataset import GNHK_Dataset
from modules.utils.auxilary_functions import *
from torch.amp import autocast, GradScaler
# from torch.cuda.amp import autocast, GradScaler
# from torch.nn import DataParallel

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)

    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary(c_classes):
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}

    index2letter = {v: k for k, v in letter2index.items()}

    return len(labels), letter2index, index2letter


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling_loader(self, model, test_loader, n, args, noise_scheduler):
        model.eval()
        
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")

        vae = vae.to(args.device)
        vae.requires_grad_(False)

        with torch.no_grad():
            pbar = tqdm(test_loader)
            for i, data in enumerate(pbar):
                images = data["latent"].to(args.device)
                text_features = data["text_emb"].to(args.device)

                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                    
                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
                
                #scheduler
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:
                    
                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

                    with torch.no_grad():
                        noisy_residual = model(x, t, text_features)
                        prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                        x = prev_noisy_sample
                    
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        
        del vae

        return x

    def sampling(self, model, vae, x_text, args, noise_scheduler, tokenizer=None, text_encoder=None):
        model.eval()
        
        with torch.no_grad():
            text_features = x_text
            n = x_text.shape[0]
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            text_features = text_encoder(**text_features).last_hidden_state    
                 
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            #scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample

            
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            x = (image.cpu().numpy() * 255).astype(np.uint8)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x



# def train(diffusion, model, ema, ema_model, optimizer, mse_loss, loader, test_loader, noise_scheduler, args, lr_scheduler=None):
#     model.train()
#     loss_meter = AvgMeter()
#     print('Training started....')
    
#     for epoch in range(args.epochs):
#         print('Epoch:', epoch)
#         pbar = tqdm(loader)
#         for data in pbar:
#             images = data[0].to(args.device)
#             text_features = data[1].to(args.device)
            
#             noise = torch.randn(images.shape).to(images.device)
#             # Sample a random timestep for each image
#             num_train_timesteps = diffusion.noise_steps
            
#             timesteps = torch.randint(
#                 0, num_train_timesteps,
#                 (images.shape[0],), device=images.device
#             ).long()
            
#             # Add noise to the clean images according to the noise magnitude
#             # at each timestep (this is the forward diffusion process)
#             noisy_images = noise_scheduler.add_noise(
#                 images, noise, timesteps
#             )
#             x_t = noisy_images
#             t = timesteps
            
#             if np.random.random() < 0.1:
#                 labels = None
            
#             predicted_noise = model(x_t, timesteps=t, context=text_features)
            
#             loss = mse_loss(noise, predicted_noise)
            
#             optimizer.zero_grad()
            
#             loss.backward()
            
#             optimizer.step()
            
#             ema.step_ema(ema_model, model)

#             count = images.size(0)
#             loss_meter.update(loss.item(), count)
#             pbar.set_postfix(MSE=loss_meter.avg)
            
#             if lr_scheduler is not None:
#                 lr_scheduler.step()
    
#         if epoch % 10 == 0:
#             labels = torch.arange(16).long().to(args.device)
#             n = len(labels)
        
#             ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, n=n, args=args, noise_scheduler=noise_scheduler)
#             epoch_n = epoch 

#             save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
            
#             torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
#             torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
#             torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt")) 



def train(diffusion, model, ema, ema_model, optimizer, mse_loss, loader, test_loader, noise_scheduler, args, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    scaler = GradScaler()  # <--- Скейлер для смешанной точности

    print('Training started....')

    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)

        for data in pbar:
            images = data["latent"].to(args.device)
            text_features = data["text_emb"].to(args.device)

            noise = torch.randn(images.shape).to(images.device)
            num_train_timesteps = diffusion.noise_steps
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            x_t = noisy_images
            t = timesteps

            if np.random.random() < 0.1:
                labels = None

            optimizer.zero_grad()

            with autocast('cuda'):  # <--- Используем FP16 в форвард-проходе
                predicted_noise = model(x_t, timesteps=t, context=text_features)
                loss = mse_loss(noise, predicted_noise)

            scaler.scale(loss).backward()  # <--- Масштабируем градиенты
            scaler.step(optimizer)  # <--- Шаг оптимизатора через скейлер
            scaler.update()  # <--- Обновляем скейлер

            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)

            if lr_scheduler is not None:
                lr_scheduler.step()

        if epoch % 10 == 0:
            labels = torch.arange(16).long().to(args.device)
            n = len(labels)

            ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, n=n, args=args, noise_scheduler=noise_scheduler)
            epoch_n = epoch

            save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)

            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))




def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='custom', help='iam, gnhk') 
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./Models/diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=True)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--stable_dif_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    parser.add_argument('--sampling_results_path', type=str, default='./sampling_results')
    parser.add_argument('--texts_to_sample_path', type=str, default='./texts.txt', help='path to txt file with texts to sample separated with newline')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    
    print('torch version', torch.__version__)
    
    #create save directories
    setup_logging(args)

    ############################ DATASET ############################
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    
    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        if args.level == 'word':
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        else:
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        print('train data', len(train_data))
        
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        
    elif args.dataset == 'gnhk':
        print('loading GNHK')
        myDataset = GNHK_Dataset
        dataset_folder = 'path/to/GNHK'
        style_classes = 515
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    
    else:

        path_h5 = '/home/fantom/Диплом/DiffusionPen/preprocessed/data.h5'
        # cyrillic_train_root = '/home/fantom/Диплом/Datasets/Cyrillic/train'
        cyrillic_train_tsv_path = '/home/fantom/Диплом/Datasets/Cyrillic/train.tsv'
        # hkr_img_dir = '/home/fantom/Диплом/Datasets/HKR/img'
        hkr_ann_dir = '/home/fantom/Диплом/Datasets/HKR/ann'
        
        full_dataset = HDF5LazyDataset(path_h5, cyrillic_train_tsv_path, hkr_ann_dir)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
        # small_dataset = torch.utils.data.Subset(full_dataset, range(1000))

        
        
        # train_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        character_classes = full_dataset.alphabet
        test_size = args.batch_size
        rest = len(full_dataset) - test_size
        train_data, test_data = random_split(full_dataset, [rest, test_size], generator=torch.Generator().manual_seed(42))
        
        

        # train_data = CustomDataset(hkr_img_dir, hkr_ann_dir, cyrillic_train_root, cyrillic_train_tsv_path, train_transform)
        # test_size = args.batch_size
        # rest = len(train_data) - test_size
        # test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        # character_classes = train_data.alphabet
    
    print('Dataset length:', len(train_data) + len(test_data))
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)


    c_classes = character_classes
    punctuation = ''.join(set(string.punctuation) & set(c_classes))
    cdict = {c:i for i,c in enumerate(c_classes)}
    icdict = {i:c for i,c in enumerate(c_classes)}

    num_char_classes, letter2index, index2letter = labelDictionary(c_classes)
    tok = False
    if not tok:
        tokens = {"PAD_TOKEN": 52}
    else:
        tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
    num_tokens = len(tokens.keys())
    print('num_tokens', num_tokens)

    print('num of character classes', num_char_classes)
    print('alphabet:', ''.join(c_classes))
    vocab_size = num_char_classes + num_tokens
    
    ######################### MODEL ####################################### 
    vocab_size = len(character_classes)
    print('Vocab size: ', vocab_size)
    
    # if args.dataparallel:
    #     device_ids = [1, 0]
    #     print('using dataparallel with device:', device_ids)
    # else:
    idx = int(''.join(filter(str.isdigit, args.device)))
    device_ids = [idx]

    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, context_dim=args.emb_dim, args=args)#.to(args.device)
    
    # if args.dataparallel:
    #     unet = DataParallel(unet) #, device_ids=device_ids)

    unet = unet.to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #load from last checkpoint
    
    if args.load_check:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', weights_only=True))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt', weights_only=True))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt', weights_only=True))
        print('Loaded models and optimizer')

    #add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    train(diffusion, unet, ema, ema_model, optimizer, mse_loss, train_loader, test_loader, ddim, args, lr_scheduler=lr_scheduler)

    
if __name__ == "__main__":
    main()
