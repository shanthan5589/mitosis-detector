import torch.nn as nn
from torchvision import transforms, models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = None
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 1)
    return model


def get_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
