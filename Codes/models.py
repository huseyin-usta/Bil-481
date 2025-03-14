import torch
import timm
from torch import nn
from torch.nn import functional as F

def create_efficientnet_model(model_name="efficientnet_b5"):
    model = timm.create_model(model_name, pretrained=True)
    
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, 2)
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    return model