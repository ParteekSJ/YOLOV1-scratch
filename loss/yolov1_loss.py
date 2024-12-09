import sys

sys.path.append("../")

import torch
import torch.nn as nn
from dataset import voc
from models import yolo
import yaml

"""
CODE TO CHECK THE OUTPUT PER CELL IN THE GRID
counter = 0
for x in range(7):
    for y in range(7):
        counter += 1
        print(f"{counter} -> {x}-{y}-{some_tensor[x, y, 0, :]}")
"""


def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    :params boxes1.shape = [1, 7, 7, 2, 4] --> [BatchSize, S, S, B, (x1,y1,x2,y2)]
    :params boxes2.shape = [1, 7, 7, 2, 4] --> [BatchSize, S, S, B, (x1,y1,x2,y2)]
    https://www.superannotate.com/blog/intersection-over-union-for-object-detection
    """
    # Area of boxes (x2-x1)*(y2-y1) = width * height
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [1, 7, 7, 2]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, 7, 7, 2]

    ## Find the intersection coordinates
    # Get the max top left x1,y1 coordinate from both bounding boxes
    x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
    y_top = torch.max(boxes1[..., 1], boxes2[..., 1])

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
    y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

    """
    > The intersection between two boxes, if it exists, will form another rectangle defined by
      top-left corner: (x_left, y_top) & Bottom-right corner: (x_right, y_bottom)
    > For 2 boxes to intersect, x_left < x_right, and y_top < y_bottom
    > If these conditions are not met, there is no overlap
    > If the intersection area is negative, it means there is no overlap between the boxes, and the IoU is 0
    """

    # intersection_area = intersecting_width * intersecting_height
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1.clamp(min=0) + area2.clamp(min=0) - intersection_area

    # IOU = Intersection Area / Union Area
    iou = intersection_area / (union + 1e-6)  # [1, 7, 7, 2]
    return iou


class YOLOV1Loss(nn.Module):
    r"""
    Loss module for YoloV1 which caters to the following components:
    1. Localization Loss for responsible predictor boxes
    2. Objectness Loss for responsible predictor boxes
    2. Objectness Loss for non-responsible predictor boxes of cells assigned with objects
    2. Objectness Loss for ALL predictor boxes of cells not assigned with objects
    3. Classification Loss
    """

    def __init__(self, S=7, B=2, C=20):
        super(YOLOV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, preds, targets, use_sigmoid=False):
        r"""
        Main method of loss computation
        :param preds: (Batch, S*S*(5B+C)) tensor
        :param targets: (Batch, S, S, (5B+C)) tensor.
            Target element for each cell has been duplicated 5B times(done in VOCDataset)
        :param use_sigmoid: Whether to use sigmoid activation for box predictions or not
        """
        batch_size = preds.size(0)

        # [Batch, S, S, 5B+C], [1, 7, 7, 30]
        preds = preds.reshape(batch_size, self.S, self.S, 5 * self.B + self.C)

        # Generally sigmoid leads to quicker convergence
        if use_sigmoid:
            preds[..., : 5 * self.B] = torch.nn.functional.sigmoid(preds[..., : 5 * self.B])

        # Shifts for all grid cell locations.
        # Will use these for converting x_center_offset/y_center_offset
        # values to x1/y1/x2/y2(normalized 0-1)

        """ 
        > Each cell in the SxS grid corresponds to a portion of the normalized image (0 to 1). 
        > shifts_x and shifts_y give the top-left corner coordinate of each cell in NORMALIZED SPACE!
        > Used to convert predicted offsets (x, y) into absolute coordinates in normalized form.
        """

        # S cells = 1 => each cell adds 1/S pixels of shift
        shifts_x = torch.arange(0, self.S, dtype=torch.int32, device=preds.device) * 1 / float(self.S)  # [7]
        shifts_y = torch.arange(0, self.S, dtype=torch.int32, device=preds.device) * 1 / float(self.S)  # [7]

        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")  # [7,7], [7,7]

        # Expanding Shifts (Creating a Grid) for Each Box Predictor -> (1, S, S, B)
        # For each bbox, we have a grid given by the last dimension. Last dim values are irrelevant.
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)  # [1, 7, 7, 2]
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)  # [1, 7, 7, 2]

        # Separate Bounding Box Predictions from Class Predictions // Extracting Prediction Subset
        # pred_boxes -> (1, S, S, 5*B) ->  (1, S, S, B, 5)
        pred_boxes = preds[..., : 5 * self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        # Shape: [1, 7, 7, 10] -> [1, 7, 7, 2, 5]

        ## Convert YOLO Format Predictions to MinMax Coordinates,
        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)

        # Retrieving the attributes of the prediction bounding box
        pred_xc = pred_boxes[..., 0]  # bbox x-centroid(s) / [1, 7, 7, 2]
        pred_yc = pred_boxes[..., 1]  # bbox y-centroid(s) / [1, 7, 7, 2]
        pred_w = pred_boxes[..., 2]  # bbox width(s) (already square-rooted) / [1, 7, 7, 2]
        pred_h = pred_boxes[..., 3]  # bbox height(s) (already square-rooted) / [1, 7, 7, 2]

        ## To convert offsets in GRID SPACE to offset in IMAGE SPACE, we do as follows:
        # x1 = (xc_offset / S - shift_x) - w
        pred_boxes_x1 = (pred_xc / self.S + shifts_x) - 0.5 * torch.square(pred_w)
        pred_boxes_x1 = pred_boxes_x1[..., None]  # [1, 7, 7, 2, 1]
        # y1 = (yc_offset / S - shift_y) - h
        pred_boxes_y1 = (pred_yc / self.S + shifts_y) - 0.5 * torch.square(pred_h)
        pred_boxes_y1 = pred_boxes_y1[..., None]  # [1, 7, 7, 2, 1]
        # x2 = (xc_offset / S + shift_y) + w
        pred_boxes_x2 = (pred_xc / self.S + shifts_x) + 0.5 * torch.square(pred_w)
        pred_boxes_x2 = pred_boxes_x2[..., None]  # [1, 7, 7, 2, 1]
        # y2 = (yc_offset / S + shift_y) - h
        pred_boxes_y2 = (pred_yc / self.S + shifts_y) + 0.5 * torch.square(pred_h)
        pred_boxes_y2 = pred_boxes_y2[..., None]  # [1, 7, 7, 2, 1]

        # Concatenating x1, x2, y1, y2 into a single tensor.
        pred_boxes_x1y1x2y2 = torch.cat(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2],
            dim=-1,
        )  # [1, 7, 7, 2, 4] - [BatchSize, S, S, B, (x1, y1, x2, y2)]

        ## Convert YOLO Format Predictions to MinMax Coordinates,
        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)

        # target_boxes -> (Batch_size, S, S, B, 5)

        target_boxes = targets[..., : 5 * self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        # Retrieving the attributes of the target bounding box
        target_xc = target_boxes[..., 0]  # bbox x-centroid
        target_yc = target_boxes[..., 1]  # bbox y-centroid
        target_w = target_boxes[..., 2]  # bbox width (already square-rooted)
        target_h = target_boxes[..., 3]  # bbox height (already square-rooted)

        target_boxes_x1 = (target_xc / self.S + shifts_x) - 0.5 * torch.square(target_w)
        target_boxes_x1 = target_boxes_x1[..., None]  # [1, 7, 7, 2, 1]

        target_boxes_y1 = (target_yc / self.S + shifts_y) - 0.5 * torch.square(target_h)
        target_boxes_y1 = target_boxes_y1[..., None]  # [1, 7, 7, 2, 1]

        target_boxes_x2 = (target_xc / self.S + shifts_x) + 0.5 * torch.square(target_w)
        target_boxes_x2 = target_boxes_x2[..., None]  # [1, 7, 7, 2, 1]

        target_boxes_y2 = (target_yc / self.S + shifts_y) + 0.5 * torch.square(target_h)
        target_boxes_y2 = target_boxes_y2[..., None]  # [1, 7, 7, 2, 1]

        # Concatenating x1, x2, y1, y2 into a single tensor.
        target_boxes_x1y1x2y2 = torch.cat(
            [target_boxes_x1, target_boxes_y1, target_boxes_x2, target_boxes_y2],
            dim=-1,
        )  # [1, 7, 7, 2, 4] - [BatchSize, S, S, B, (x1, y1, x2, y2)]

        ## Compute IOU
        """
        > For each cell, each of the B predicted boxes is compared with the corresponding ground truth box,
        and the IoU (Intersection over Union) is calculated.
        > Box with the highest IoU for that cell is considered the "responsible" predictor.
        """
        # iou -> (Batch_size, S, S, B) // [1, 7, 7, 2]
        iou = get_iou(pred_boxes_x1y1x2y2, target_boxes_x1y1x2y2)

        # max_iou_val/max_iou_idx -> (Batch_size, S, S, 1)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)  # [1, 7, 7, 1], [1, 7, 7, 1]

        #########################
        # Indicator Definitions #
        #########################
        # before max_iou_idx -> (Batch_size, S, S, 1) Eg [[0], [1], [0], [0]]
        # after repeating max_iou_idx -> (Batch_size, S, S, B)
        # Eg. [[0, 0], [1, 1], [0, 0], [0, 0]] assuming B = 2
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)  # [1, 7, 7, 2]
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx).to(preds.device)
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()  # [1, 7, 7, 2]

        # obj_indicator -> (Batch_size, S, S, 1)
        obj_indicator = targets[..., 4:5]  # 1 if object is present in that cell, else 0

        # Loss definitions start from here

        #######################
        # Classification Loss #
        #######################
        cls_target = targets[..., 5 * self.B :]  # [1, 7, 7, 20]
        cls_preds = preds[..., 5 * self.B :]  # [1, 7, 7, 20]
        cls_mse = (cls_preds - cls_target) ** 2
        # Only keep losses from cells with object assigned
        cls_mse = (obj_indicator * cls_mse).sum()

        ######################################################
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        # indicator is now object_cells * is_best_box
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator  # [1, 7, 7, 2] * [1, 7, 7, 1]
        obj_mse = (pred_boxes[..., 4] - max_iou_val) ** 2  # [1, 7, 7, 2] - [1, 7, 7, 1] = [1, 7, 7, 2]
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        #####################
        # Localization Loss #
        #####################
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor

        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2  # [1, 7, 7, 2]
        x_mse = (is_max_box_obj_indicator * x_mse).sum()
        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2  # [1, 7, 7, 2]
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2  # [1, 7, 7, 2]
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
        h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2  # [1, 7, 7, 2]
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()

        #################################################
        # Objectness Loss
        # For boxes of cells assigned with object that
        # aren't responsible predictor boxes
        # and for boxes of cell not assigned with object
        #################################################
        no_object_indicator = 1 - is_max_box_obj_indicator  # [1, 7, 7, 2]
        no_obj_mse = (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()

        ##############
        # Total Loss #
        ##############
        loss = self.lambda_coord * (x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj * no_obj_mse
        loss = loss / batch_size
        return loss


if __name__ == "__main__":
    # Loading the configuration
    config_path = "../config/voc.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Initializing a model with the configuration parameters
    model = yolo.YOLOV1(im_size=224, num_classes=20, model_config=config["model_params"])
    # Creating a random tensor
    random_image = torch.randn(1, 3, 448, 448)
    # Performing Inference
    with torch.inference_mode():
        op = model(random_image)

    # Creating an instance of the dataset and retrieving an item
    voc_dataset = voc.VOCDataset(
        split="train",
        im_sets=["/Users/parteeksj/Desktop/VOCdevkit/VOC2007/"],
    )
    ground_truth = voc_dataset.__getitem__(0)[1]["yolo_targets"]

    # Loss
    criterion = YOLOV1Loss()
    loss = criterion(op, ground_truth.unsqueeze(0).float())
    print(f"{loss=}")
