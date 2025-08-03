import torch

class Config:
    # Data
    input_size = (32, 256)
    # charset = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789!?,.- "
    charset = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!?,.- "
    kzcharset = "әғқңөұүһӘҒҚҢӨҰҮҺ"
    train_ratio = 0.95
    
    # Model
    cnn_out_channels = 256
    rnn_hidden_size = 256
    rnn_num_layers = 2

    # Training
    batch_size = 128
    lr = 0.001
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
config = Config()