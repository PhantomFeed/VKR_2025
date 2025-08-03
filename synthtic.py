from PIL import Image, ImageDraw, ImageFont
import random
import os
from config import Config

class SyntheticTextGenerator:
    def __init__(self, fonts_dir='data/fonts/', output_size=(256, 64), debug=False):
        self.fonts = self._load_fonts(fonts_dir)
        self.output_size = output_size
        self.charset = Config.charset
        # self.current_font = None  

    def _load_fonts(self, fonts_dir):
        fonts = []
        for font_file in os.listdir(fonts_dir):
            if font_file.endswith(('.ttf', '.otf')):
                try:
                    font_path = os.path.join(fonts_dir, font_file)
                    fonts.append(font_path)
                except:
                    continue
        return fonts

    def _generate_text(self):
        length = random.randint(4, 12)
        return ''.join(random.choice(self.charset) for _ in range(length))

    def generate_sample(self):
        font_path = random.choice(self.fonts)
        # self.current_font = os.path.basename(font_path) 
        font_size = random.randint(14, 26)
        text = self._generate_text()
        
        # Создаем временное изображение для измерения текста
        temp_img = Image.new('L', (1, 1), 255)
        temp_draw = ImageDraw.Draw(temp_img)
        font = ImageFont.truetype(font_path, font_size)
        
        # Получаем ограничивающий прямоугольник текста
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Создаем основное изображение
        img = Image.new('L', (self.output_size[0], self.output_size[1]), color=255)
        draw = ImageDraw.Draw(img)
        
        # Координаты для центрированного текста
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
        
        # Рисуем текст
        draw.text((x, y), text, font=font, fill=0)
        
        # Масштабируем
        # img = img.resize(self.output_size, Image.BILINEAR)
        
        return img, text
        # , self.current_font  