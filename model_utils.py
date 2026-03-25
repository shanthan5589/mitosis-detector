import torch.nn as nn
from torchvision import transforms, models

IMAGE_SIZE = None
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_SLIDES =  [4, 12, 13, 15, 17, 19, 21, 22, 24, 25, 26, 28, 29, 32, 34, 35, 36]
VAL_SLIDES = [7, 8, 14, 23]

def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
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
        transforms.Normlize(mean=MEAN, std=STD)
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])