import numpy as np
from torchvision import transforms

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255.0
ORIGINAL_IMAGE_SIZE = (640, 360)

# Define the transforms for training and testing datasets
train_transform = transforms.Compose([
    transforms.Resize(ORIGINAL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
test_transform = transforms.Compose([
    transforms.Resize(ORIGINAL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
target_transform = transforms.Compose([
    transforms.Resize(ORIGINAL_IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
])
