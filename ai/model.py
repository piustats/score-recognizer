import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Model(torch.nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        # Load Faster RCNN pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=None

        )

        # Get the number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
