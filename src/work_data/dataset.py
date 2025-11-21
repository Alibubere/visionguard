import torch
from torch.utils.data import Dataset
from PIL import Image
import os 
import json 
import numpy as np

class COCOMergedDataset(Dataset):
    def __init__(self, images_list , annotations_list , img_dir , transforms=None):
        
        self.images = images_list
        self.img_dir = img_dir
        self.transforms = transforms
        self.ann_index = {}

        for ann in annotations_list:
            img_id = ann["image_id"]
            if img_id not in self.ann_index:
                self.ann_index[img_id] = []

            self.ann_index.setdefault(img_id,[]).append(ann)

    def __len__(self):
        return len(self.images)
    
