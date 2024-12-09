import torch
import torch.nn as nn
import torchvision
import yaml


class YOLOV1(nn.Module):
    r"""
    Model with three components:
    1. Backbone of resnet34 pretrained on 224x224 images from Imagenet
    2. 4 Conv,Batchnorm,LeakyReLU Layers for Yolo Detection Head
    3. Fc layers with final layer having S*S*(5B+C) output dimensions
        Final layer predicts [
            x_offset_box1,y_offset_box1,sqrt_w_box1,sqrt_h_box1,conf_box1, # box-1 params
            ...,
            x_offset_boxB,y_offset_boxB,sqrt_w_boxB,sqrt_h_boxB,conf_boxB, # box-B params
            p1, p2, ...., pC-1, pC  # class conditional probabilities
        ] for each S*S grid cell
    """

    def __init__(self, im_size, num_classes, model_config):
        super(YOLOV1, self).__init__()

        self.im_size = im_size
        self.im_channels = model_config["im_channels"]
        self.backbone_channels = model_config["backbone_channels"]
        self.yolo_conv_channels = model_config["yolo_conv_channels"]
        self.conv_spatial_size = model_config["conv_spatial_size"]
        self.leaky_relu_slope = model_config["leaky_relu_slope"]
        self.yolo_fc_hidden_dim = model_config["fc_dim"]
        self.yolo_fc_dropout_prob = model_config["fc_dropout"]
        self.use_conv = model_config["use_conv"]
        self.S = model_config["S"]
        self.B = model_config["B"]
        self.C = num_classes

        backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)

        ###################
        # Backbone Layers #
        ###################
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )  # discarding the FC layers from the pretrained model.

        #########################################
        # Detection Conv Layers (4 CONV LAYERS)#
        ########################################
        self.conv_yolo_layers = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.yolo_conv_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels, self.yolo_conv_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels, self.yolo_conv_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels, self.yolo_conv_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
        )

        ####################
        # Detection Layers #
        ####################
        if self.use_conv:
            self.fc_yolo_layers = nn.Sequential(
                nn.Conv2d(self.yolo_conv_channels, 5 * self.B + self.C, 1),
            )
        else:
            self.fc_yolo_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    self.conv_spatial_size * self.conv_spatial_size * self.yolo_conv_channels,
                    self.yolo_fc_hidden_dim,
                ),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.yolo_fc_dropout_prob),
                nn.Linear(self.yolo_fc_hidden_dim, self.S * self.S * (5 * self.B + self.C)),
            )

    def forward(self, x):
        out = self.features(x)  # [1, 3, 448, 448] ->  [1, 512, 14, 14]
        out = self.conv_yolo_layers(out)  # [1, 512, 14, 14] -> [1, 1024, 7, 7]
        out = self.fc_yolo_layers(out)  # [1, 1024, 7, 7] ->  [1, 30, 7, 7]
        if self.use_conv:
            # Reshape conv output to Batch x S x S x (5B+C)
            out = out.permute(0, 2, 3, 1)  # [1, 30, 7, 7] -> [1, 7, 7, 30]
        return out


if __name__ == "__main__":
    config_path = "../config/voc.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = YOLOV1(im_size=224, num_classes=20, model_config=config["model_params"])
    random_image = torch.randn(1, 3, 448, 448)
    with torch.inference_mode():
        op = model(random_image)
