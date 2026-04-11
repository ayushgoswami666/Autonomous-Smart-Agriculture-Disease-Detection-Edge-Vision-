import torch
import torch.nn as nn
from torchvision import models
import os

def load_model():

    model = models.vgg16(weights=None)

    # Absolute path of this file
    base_dir = os.path.dirname(__file__)

    model_path = os.path.join(base_dir, "plant_disease_vgg16.pth")

    state_dict = torch.load(model_path, map_location="cpu")

    num_classes = state_dict['classifier.6.weight'].shape[0]
    print(f"Number of classes: {num_classes}")

    model.classifier[6] = nn.Linear(4096, num_classes)

    model.load_state_dict(state_dict)

    model.eval()

    return model