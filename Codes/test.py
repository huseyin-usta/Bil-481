from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
from model import *

class TestDataset(Dataset):
    def __init__(self, csv, transform=None):
        df = pd.read_csv(csv)
        self.csv = csv
        self.transform = transform
        self.image_paths = list(df.get("id"))
        print(f"Dataset loaded with {len(self)} images.")
    def __getitem__(self, index):
        path = f"dataset/{self.image_paths[index]}"
        image = Image.open(path).convert("RGB")
        image = numpy.array(image)
        if self.transform:
            image = self.transform(image=image)
        return image, self.image_paths[index]
    
    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    for w in [31]:
        model = create_model_special()
        checkpoint = torch.load(f"model_saves/special/weights_at_{w}.pth")
        model.load_state_dict(checkpoint)

        model.to("cuda")

        test_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        test_set = TestDataset("dataset/test.csv", test_transform)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16)

        ids = []
        predictions = []
        with torch.no_grad():
            for inputs, image_ids in test_loader:
                inputs = inputs['image'].to("cuda")
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                preds = preds.to("cpu")
                for i, img_id in enumerate(image_ids):
                    predictions.append(preds[i].item())
                    ids.append(img_id)

        df = pd.DataFrame({
            'id': ids,
            'label': predictions
        })

        df.to_csv(f'special_{w}_.csv', index=False)