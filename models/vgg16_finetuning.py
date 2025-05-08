import torch.nn as nn
from torchvision import models

def get_vgg16_finetuning(num_classes):
    
    vgg16 = models.vgg16(pretrained=True)

    # first 5 layers (1.conv block) freeze
    for param in vgg16.features[:5].parameters():
        param.requires_grad = False

    # classifier change
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return vgg16


