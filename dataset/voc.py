import os
import albumentations as albu
import cv2
import torch
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from pprint import pprint
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

"""
1, Retrieve all filenames from trainval.txt from ImageSets/Main folder.
2, 
"""


def load_images_and_anns(im_sets, label2idx, ann_fname, split):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: Sets of images to consider
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :return:
    """
    im_infos = []

    for im_set in im_sets:
        im_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(im_set, "ImageSets", "Main", f"{ann_fname}.txt")):
            im_names.append(line.strip())

        # Set annotation and image path
        ann_dir = os.path.join(im_set, "Annotations")  # './VOCdevkit/VOC2007/Annotations'
        im_dir = os.path.join(im_set, "JPEGImages")  # './VOCdevkit/VOC2007/JPEGImages'

        # Iterating over all images and retrieving metadata.
        for im_name in im_names:
            # Retrieving the annotation XML file for a specific image
            ann_file = os.path.join(ann_dir, f"{im_name}.xml")
            # Initializing an empty dictionary to store image metadata.
            im_info = {}
            # parsing the XML file. Returns an XML tree.
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find("size")
            # Retrieving the image width and height
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            im_info["img_id"] = os.path.basename(ann_file).split(".xml")[0]  # img_id
            im_info["filename"] = os.path.join(im_dir, "{}.jpg".format(im_info["img_id"]))  # filepath
            im_info["width"] = width
            im_info["height"] = height
            detections = []

            # We will keep an image only if there are valid rois in it
            any_valid_object = False

            # Retrieving the ROI (x1, y1, x2, y2)
            for obj in ann_info.findall("object"):
                det = {}  # Initializing an empty detection dictionary
                label = label2idx[obj.find("name").text]  # convert label to corresponding index
                difficult = int(obj.find("difficult").text)
                bbox_info = obj.find("bndbox")
                # Retrieving the BBOX coordinates - [x_min, y_min, x_max, y_max]
                # PASCAL VOC uses a 1-based indexing for bounding box coordinates. hence, "-1"
                bbox = [
                    int(float(bbox_info.find("xmin").text)) - 1,
                    int(float(bbox_info.find("ymin").text)) - 1,
                    int(float(bbox_info.find("xmax").text)) - 1,
                    int(float(bbox_info.find("ymax").text)) - 1,
                ]
                # Filling in the detection dictionary
                det["label"] = label
                det["bbox"] = bbox
                det["difficult"] = difficult
                # Ignore difficult rois during training
                # At test time eval does the job of ignoring difficult
                # examples.
                if difficult == 0 or split == "test":
                    detections.append(det)
                    any_valid_object = True

            if any_valid_object:
                # For each image a detection key is made.
                im_info["detections"] = detections
                im_infos.append(im_info)

    # im_info has the following keys - ['img_id', 'filename', 'width', 'height', 'detections']
    print("Total {} images found".format(len(im_infos)))
    return im_infos


class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size: int = 448, S: int = 7, B: int = 2, C: int = 20):
        super().__init__()

        self.split = split

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = "trainval" if self.split == "train" else "test"
        self.im_size = im_size

        # Grid Size
        self.S = S
        self.B = B
        self.C = C

        # Train and test transformations.
        # "Also bounding box and keypoint transformations if the appropriate parameters are provided"
        self.transforms = {
            "train": albu.Compose(
                [
                    albu.HorizontalFlip(p=0.5),
                    albu.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), always_apply=True),
                    albu.ColorJitter(
                        brightness=(0.8, 1.2),
                        contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2),
                        hue=(-0.2, 0.2),
                        always_apply=None,
                        p=0.5,
                    ),
                    albu.Resize(self.im_size, self.im_size),
                ],
                bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"]),
            ),
            "test": albu.Compose(
                [
                    albu.Resize(self.im_size, self.im_size),
                ],
                bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"]),
            ),
        }

        # Convert an image to tensor, and normalize it.
        # https://pytorch.org/hub/pytorch_vision_resnet/
        self.tensor_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        classes = [
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        ]
        classes = sorted(classes)

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        # pprint(self.idx2label)

        # Retrieve all image&bbox related data from `load_images_and_anns`
        self.images_info = load_images_and_anns(
            im_sets=self.im_sets,
            label2idx=self.label2idx,
            ann_fname=self.fname,
            split=self.split,
        )

        """
        Example of Parsed Image with Annotations, i.e., 
        >> self.images_info[0] - 5 keys
        {'detections': [{'bbox': [262, 210, 323, 338], 'difficult': 0, 'label': 8},
                        {'bbox': [164, 263, 252, 371], 'difficult': 0, 'label': 8},
                        {'bbox': [240, 193, 294, 298], 'difficult': 0, 'label': 8}],
         'filename': '/Users/parteeksj/Desktop/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
         'height': 375,
         'img_id': '000005',
         'width': 500}
        """

    def __len__(self):
        return len(self.images_info)

    def get_annotated_image(self, image, bboxes, labels):
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.Tensor(bboxes)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))

        annotated_image = draw_bounding_boxes(
            image=image,
            boxes=bboxes,
            colors="red",
            width=2,
            labels=[str(x) for x in labels],
        )

        return annotated_image

    def __getitem__(self, index):
        im_info = self.images_info[index]  # Retrieve the corresponding im_info element.
        im = cv2.imread(im_info["filename"])  # [H, W, C], (0, 255)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # [H, W, C], (0, 255)

        # Get annotations for this image
        bboxes = [detection["bbox"] for detection in im_info["detections"]]  # (0, 448)
        labels = [detection["label"] for detection in im_info["detections"]]
        difficult = [detection["difficult"] for detection in im_info["detections"]]

        # pre_transform_annotated_image = self.get_annotated_image(im, bboxes, labels)

        # Transform Image & Bounding Boxes according to the transforms.
        transformed_info = self.transforms[self.split](image=im, bboxes=bboxes, labels=labels)
        im = transformed_info["image"]  # [H, W, C], (0, 255)
        bboxes = torch.as_tensor(transformed_info["bboxes"])  # (0, 448)
        labels = torch.as_tensor(transformed_info["labels"])
        difficult = torch.as_tensor(difficult)

        # post_transform_annotated_image = self.get_annotated_image(im, bboxes, labels)

        # Convert image to tensor and normalize
        im_tensor = self.tensor_transforms(im)  # [C, H, W] // [3, 448, 448]
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)

        # Build Target for Yolo, i.e., (S x S x [5 + C])
        target_dim = 5 * self.B + self.C  # 5B + C
        h, w = im.shape[:2]  # retrieving image height, width
        yolo_targets = torch.zeros(self.S, self.S, target_dim)  # [j(y), i[x], 30] // [7, 7, 30]

        # Height and width of grid cells is H // S
        cell_pixels = h // self.S  #  448 // 7 = 64 --> num pixels in a single cell.

        if len(bboxes) > 0:
            # Convert x1y1x2y2 to xywh format [YOLO format]
            box_widths = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]  # w = (x2 - x1)
            box_heights = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]  # h = (y2 - y1)
            box_center_x = bboxes_tensor[:, 0] + 0.5 * box_widths  # center_x = (x1 + 0.5w)
            box_center_y = bboxes_tensor[:, 1] + 0.5 * box_heights  # center_y = (y1 + 0.5h)

            # Get cell i,j from xc, yc (bbox centroids), i.e., cell where the bbox centroid falls.
            box_i = torch.floor(box_center_x / cell_pixels).long()
            box_j = torch.floor(box_center_y / cell_pixels).long()

            # x_center, y_center offset from top left cell [box_i, box_j] coordinate [NORMALIZED]
            box_xc_cell_offset = (box_center_x - box_i * cell_pixels) / cell_pixels
            box_yc_cell_offset = (box_center_y - box_j * cell_pixels) / cell_pixels

            # w, h targets normalized to 0-1
            box_w_label = box_widths / w  # dividing by image width
            box_h_label = box_heights / h  # dividing by image height

            # Iterate through all the bboxes for this specific image.
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                # Make target of the exact same shape as prediction
                for k in range(self.B):
                    s = 5 * k
                    # target_ij = [xc_offset,yc_offset,sqrt(w),sqrt(h), conf, cls_label]
                    yolo_targets[box_j[idx], box_i[idx], s] = box_xc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s + 1] = box_yc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s + 2] = box_w_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s + 3] = box_h_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s + 4] = 1.0  # as the bbox contains the obj
                label = int(labels[b])

                # Create a one-hot label distribution for the ground truth target distribution
                cls_target = torch.zeros((self.C,))  # [20,] filled with 0s
                cls_target[label] = 1.0
                # Appending the class distribution to the yolo target vector
                yolo_targets[box_j[idx], box_i[idx], 5 * self.B :] = cls_target

        # For training, we use yolo_targets(xoffset, yoffset, sqrt(w), sqrt(h))
        # For evaluation we use bboxes_tensor (x1, y1, x2, y2)
        # Below we normalize bboxes tensor to be between 0-1
        # as thats what evaluation script expects so (x1/w, y1/h, x2/w, y2/h)
        if len(bboxes) > 0:
            bboxes_tensor /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes_tensor)
        targets = {
            "bboxes": bboxes_tensor,  # (0, 1)
            "labels": labels_tensor,
            "yolo_targets": yolo_targets,  # [S, S, 5B+C] all between (0, 1)
            "difficult": difficult,
        }
        return im_tensor, targets, im_info["filename"]


if __name__ == "__main__":
    voc_dataset = VOCDataset(
        split="train",
        im_sets=["/Users/parteeksj/Desktop/VOCdevkit/VOC2007/"],
    )

    _x = voc_dataset.__getitem__(0)
