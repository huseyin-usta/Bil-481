import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
import torch
import os
from train import train_model
from dataset import TrainDataset
from torch.utils.data import DataLoader

def fine_tune(csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "fine_tuned_model.pth" if os.path.exists("fine_tuned_model.pth") else "model.pth"

    train_transform = A.Compose([
        A.Resize(232, 232),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.1, border_mode=0, p=0.4),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1),
                A.HueSaturationValue(p=1),
                A.GridDropout(ratio=0.4, unit_size_min=2, unit_size_max=3, random_offset=True, p=1),
            ],
            p=0.7,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
                A.GaussNoise(var_limit=(10.0, 60.0), p=1),
            ],
            p=0.7,
        ),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.RandomCrop(224, 224, 1),
        ToTensorV2()
    ])

    train_set = TrainDataset(csv, train_transform)
    train_loader = DataLoader(train_set, batch_size=8, num_workers=2, shuffle=True)
    model = torch.load(model_path, weights_only=False) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_model(model, optimizer, criterion, train_loader, device)