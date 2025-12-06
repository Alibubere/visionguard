import torch 
from torchvision.transforms import functional as f
from PIL import Image
import json 
import logging
from src.model import get_model


def load_categories(categories_json_path):
    
    with open(categories_json_path,"r") as f:
        data = json.load(f)

    cats = data["categories"]

    id_to_name = {c["id"]: c["name"] for c in cats}

    return id_to_name , len(cats)


def load_trained_model(checkpoint_path , categories_json_path, device):

    id_to_name,num_cats = load_categories(categories_json_path)

    num_classes = num_cats +1 

    model = get_model(num_classes=num_classes)
    checkpoint = torch.load(f=checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    logging.info(f"Loaded model from {checkpoint_path} with {num_classes} classes")

    return model , id_to_name



