import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

def build_model(num_materials, num_qualities, device):
    base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(base.children())[:-1]).to(device)
    f_dim = base.classifier[1].in_features
    mat_head = nn.Linear(f_dim, num_materials).to(device)
    qual_head = nn.Linear(f_dim, num_qualities).to(device)
    return features, mat_head, qual_head
