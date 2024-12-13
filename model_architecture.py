import torch.nn as nn
import torch

class ConvNet(nn.Module):
    """
    Convolutional Neural Network model for image classification on endangered species.
    4 convolutional layers and 3 fully connected layers.
        - More layers to better caputure

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.fully_connection_layers = nn.Sequential(
           
            nn.Dropout(0.4),   # increased dropout
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),  # Added batch normalization
            nn.ReLU(inplace=True),
           
            nn.Dropout(0.4),  # increased dropout
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),  # added batch normalization
            nn.ReLU(inplace=True),
           
            nn.Dropout(0.4),   # increased dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # added batch normalization
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 11),
        )
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connection_layers(x)
        return x