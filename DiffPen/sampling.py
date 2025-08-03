import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from modules.models.unet import UNetModel
from torchvision import transforms
from modules.models.feature_extractor import ImageEncoder
# from modules.utils.custom_dataset import CustomDataset
from modules.utils.iam_dataset import IAMDataset
from modules.utils.GNHK_dataset import GNHK_Dataset
from modules.utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)


def labelDictionary(c_classes):
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}

    return len(labels), letter2index, index2letter


def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


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

    def sampling_loader(self, model, test_loader, vae, n, args, noise_scheduler, tokenizer=None, text_encoder=None):
        model.eval()
        
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]

                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                text_features = text_encoder(**text_features).last_hidden_state  
            
                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                    
                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
                
                #scheduler
                noise_scheduler.set_timesteps(1000)
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
            image = vae.module.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

    def sampling(self, model, vae, x_text, args, noise_scheduler, tokenizer=None, text_encoder=None, batch_size=4):
        model.eval()
        imgs = []
        for start in tqdm(range(0, len(x_text), batch_size)): 
            with torch.no_grad():
                text_features = x_text[start:start+batch_size]
                n = len(text_features)
                text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
                text_features = text_encoder(**text_features).last_hidden_state    
                    
                if args.latent == True:
                    x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)      
                else:
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
                image = vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)
                x = (image.cpu().numpy() * 255).astype(np.uint8)

            else:
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
            
            imgs.append(x)
        
        imgs = np.concatenate(imgs, axis=0)
        return imgs
    

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))   
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./Models/diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
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

    character_classes = r'!$%()+,-./0123456789:;?@АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё'
    c_classes = character_classes

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
    
    # if args.dataparallel==True:
    #     device_ids = [3,4]
    #     print('using dataparallel with device:', device_ids)
    # else:
    idx = int(''.join(filter(str.isdigit, args.device)))
    device_ids = [idx]

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    # text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)
    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, context_dim=args.emb_dim, args=args)#.to(args.device)
    
    # unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)

    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.latent:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        # vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    #add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    if not os.path.exists(args.sampling_results_path):
        os.mkdir(args.sampling_results_path)

    with open(args.texts_to_sample_path) as f:
        x_text = [x.strip() for x in f.readlines()][:]
    
    print('Sampling started....')
    
    unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
    print('unet loaded')
    unet.eval()
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt', map_location=args.device, weights_only=True))
    ema_model.eval()
    
    ############################ SAMPLING ##############################
    ema_sampled_images = diffusion.sampling(ema_model, vae, x_text=x_text, args=args, noise_scheduler=ddim, tokenizer=tokenizer, text_encoder=text_encoder, batch_size=args.batch_size)  
    for i, img in enumerate(np.transpose(ema_sampled_images, axes=(0, 2, 3, 1))):
        cv2.imwrite(os.path.join(args.sampling_results_path, f'9{i}_{x_text[i]}.png'), img)

    
if __name__ == "__main__":
    main()