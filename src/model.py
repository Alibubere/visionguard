from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch

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

    model.train()  # IMPORTANT

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
