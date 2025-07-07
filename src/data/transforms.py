import numpy as np
from torchvision import transforms

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
ADE_STD  = np.array([58.395,  57.120,  57.375]) / 255.0

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

target_transform = transforms.Compose([
    # Add mask transforms if needed
])
