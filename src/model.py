from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import logging

def get_model(num_classes):

    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layers = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layers,
        num_classes
    )

    model.train()

    return model

def freeze_backbone(model):
    
    for param in model.backbone.parameters():
        param.requires_grad = False

    return model 


def unfreeze_backbone(model):

    for param in model.backbone.parameters():
        param.requires_grad = True

    return model

def get_optimizer(model, lr, weight_decay):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    return optimizer


def get_lr_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[16, 22],
        gamma=0.1
    )
    return scheduler


def save_checkpoint(model, optimizer, epoch, path):

    checkpoint ={
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch":epoch
    }

    torch.save(checkpoint,path)

    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path,device="cuda"):

    checkpoint = torch.load(path,map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch",1)+1

    logging.info(f"Checkpoint loaded from {path}, resuming at epoch {start_epoch}")
    return model , optimizer , start_epoch

