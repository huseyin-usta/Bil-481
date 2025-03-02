from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, csv, transform=None):
        df = pd.read_csv(csv)
        self.csv = csv
        self.transform = transform
        self.image_paths = list(df.get("file_name"))
        self.labels = list(df.get("label"))
        print(f"Dataset loaded with {len(self)} images.")
    def __getitem__(self, index):
        path = f"dataset/{self.image_paths[index]}"
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)
        return image, self.labels[index]
    
    def __len__(self):
        return len(self.labels)
