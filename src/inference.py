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

