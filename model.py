import torch.nn as nn
from config import Config

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class CRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            ResBlock(64, 64, stride=(2, 1)),
            ResBlock(64, 128, stride=2),  
            ResBlock(128, 128, stride=(2, 1)),
            ResBlock(128, 256, stride=2),  
            ResBlock(256, 256, stride=(2, 1)),
            # nn.AdaptiveAvgPool2d((1, None))  # [256, 1, 64]
        )


            
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            bidirectional=False,
            batch_first=False
        )

        
        self.fc = nn.Linear(config.rnn_hidden_size, len(config.charset) + 1) 
        
    def forward(self, x):
        # CNN
        x = self.cnn(x)  # [B, C=256, H=1, W=64]
        x = x.squeeze(2)  # [B, C, W]
        x = x.permute(2, 0, 1)  # [W, B, C]
        
        # RNN
        out, _ = self.rnn(x)

        # FC
        out = self.fc(out)  # [W, B, num_classes]
        return out