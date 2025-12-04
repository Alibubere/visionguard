import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as maskUtils
import torchvision.transforms.functional as f
import os
import json
import numpy as np


class COCOMergedDataset(Dataset):
    def __init__(self, images_list, annotations_list, img_dir, transforms=None):

        self.images = images_list
        self.img_dir = img_dir
        self.transforms = transforms
        self.ann_index = {}

        for ann in annotations_list:
            img_id = ann["image_id"]
            if img_id not in self.ann_index:
                self.ann_index[img_id] = []

            self.ann_index.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.ann_index.get(img_id, [])

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])

            labels.append(ann["category_id"])

            seg = ann["segmentation"]
            rles = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
            mask = maskUtils.decode(rles).astype(np.uint8)

            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)

            masks.append(mask)

            area = ann.get("area", w * h)
            areas.append(area)

            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        if len(masks) > 0:
            masks_np = np.stack(masks, axis=0)  # (N, H, W)
            masks = torch.as_tensor(masks_np, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)


        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        if self.transforms:
            img, target = self.transforms(img, target)

        else:
            img = f.to_tensor(img)
        
        return img, target
