import torch
import timm
from torch import nn
from torch.nn import functional as F

# EfficientNet ailesinden istenilen modeli imagenet ağırlıkları ile başlatır ve sonuna dropout+linear ekler.
def create_efficientnet_model(model_name="efficientnet_b5"):
    model = timm.create_model(model_name, pretrained=True)
    
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, 2)
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    return model

# ConvNext Base modelini imagenet ağırlıkları ile başlatır ve sonuna özel bir classifier katmanı (head) ekler.
def create_convnext(num_classes=2, freeze_at_beginning=True):
    model = timm.create_model('convnext_base', pretrained=True)
    
    # Backbone dışındaki (head dışındaki) tüm parametreleri donduruyoruz
    if freeze_at_beginning:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
    if hasattr(model, 'global_pool'):
        model.global_pool = nn.Identity()
    
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    return model