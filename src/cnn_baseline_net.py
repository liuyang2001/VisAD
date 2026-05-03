import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    def __init__(self, num_channels=6, matrix_size=27):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_channels, kernel_size=2, stride=2),
            nn.Sigmoid() 
        )
        self.upsample = nn.Upsample(size=(matrix_size, matrix_size), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x