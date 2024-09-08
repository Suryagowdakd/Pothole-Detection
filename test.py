import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import seaborn as sns
import copy
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torchvision
from torchvision.models.detection.ssd import SSDHead, det_utils
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import torchvision.transforms.functional as tf
import albumentations as A
import pycocotools
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision

def parse_xml(annot_path):
    tree = ET.parse(annot_path)
    root = tree.getroot()
    
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    boxes = []
    
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        
    return boxes, height, width

img_dir = "C:/Users/katty/Downloads/archive(4)/images"
annot_dir = "C:/Users/katty/Downloads/archive(4)/annotations"

# label 0 is fixed for background
classes = ["background", "pothole"]

num_classes = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
epochs = 10
learning_rate = 3e-5

model_weights_file = "model.pth"

threshold = 0.25
iou_threshold = 0.75

ignore_img = []
for annot_name in os.listdir(annot_dir):
    img_name = annot_name[:-4] + ".png"
    annot_path = os.path.join(annot_dir, annot_name)
    boxes, height, width = parse_xml(annot_path)
    
    for box in boxes:
        if box[0] < 0 or box[0] >= box[2] or box[2] > width:
            print(box[0], box[2], width)
            print("x", annot_name)
            print("*" * 50)
            ignore_img.append(img_name)
        elif box[1] < 0 or box[1] >= box[3] or box[3] > height:
            print(box[1], box[3], height)
            print("y", img_name)
            print("*" * 50)
            ignore_img.append(img_name)

# Data Augmentation

train_transform = A.Compose([A.HorizontalFlip(),
                           A.ShiftScaleRotate(rotate_limit=15, value=0,
                                              border_mode=cv2.BORDER_CONSTANT),

                           A.OneOf(
                                   [A.CLAHE(),
                                    A.RandomBrightnessContrast(),
                                    A.HueSaturationValue()], p=1),
                           A.GaussNoise(),
                           A.RandomResizedCrop(height=480, width=480)],
                          bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.15,
                                                   label_fields=["labels"]))
                           
val_transform = A.Compose([A.Resize(height=480, width=480)],
                        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.15,
                                                 label_fields=["labels"]))

class PotholeDetection(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.img_list = sorted([img for img in os.listdir(self.img_dir) 
                              if img not in ignore_img])
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annot_name = img_name[:-4] + ".xml"
        annot_path = os.path.join(self.annot_dir, annot_name)
        boxes, height, width = parse_xml(annot_path)
        labels = [1] * len(boxes)
        
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
        
        if len(np.array(boxes).shape) != 2 or np.array(boxes).shape[-1] != 4:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
                
        img = img / 255
        img = tf.to_tensor(img)
        img = img.to(dtype=torch.float32)
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["id"] = torch.tensor(idx)
            
        return img, target
    
train_ds = PotholeDetection(img_dir, annot_dir, train_transform)
val_ds = PotholeDetection(img_dir, annot_dir, val_transform)

idxs = list(range(len(train_ds)))

np.random.shuffle(idxs)
train_idx = idxs[:int(0.85 * len(train_ds))]
val_idx = idxs[int(0.85 * len(train_ds)):]

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)

def show_bbox(img, target, color=(0, 255, 0)):
    img = np.transpose(img.cpu().numpy(), (1, 2, 0))
    boxes = target["boxes"].cpu().numpy().astype("int")
    labels = target["labels"].cpu().numpy()
    img = img.copy()
    for i, box in enumerate(boxes):
        idx = int(labels[i])
        text = classes[idx]

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
        cv2.putText(img, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img

def collate_fn(batch):
    return tuple(zip(*batch))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
                    collate_fn=collate_fn,
                    pin_memory=True if device == "cuda" else False)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                  collate_fn=collate_fn,
                  pin_memory=True if device == "cuda" else False)

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

in_channels = det_utils.retrieve_out_channels(model.backbone, (480, 480))
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors,
                   num_classes=num_classes)

model.to(device)

for params in model.backbone.features.parameters():
    params.requires_grad = False
    
parameters = [params for params in model.parameters() if params.requires_grad]

optimizer = optim.Adam(parameters, lr=learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                  patience=7, threshold=0.0001)

def get_lr(optimizer):
    for params in optimizer.param_groups:
        return params["lr"]

loss_history = {
    "training_loss": [],
    "validation_loss": []
}

train_len = len(train_dl.dataset)
val_len = len(val_dl.dataset)

best_validation_loss = np.inf
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0
    
    current_lr = get_lr(optimizer)
    
    # During training, the model expects both the input tensors, as well as the targets 
    model.train()
    for imgs, targets in train_dl:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for (k, v) in d.items()} for d in targets]
        
        loss_dict = model(imgs, targets)
        losses = sum(loss_dict.values())
        training_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    with torch.no_grad():
        for imgs, targets in val_dl:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for (k, v) in d.items()} for d in targets]
            
            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())
            validation_loss += losses.item()
        
    lr_scheduler.step(validation_loss)
    if current_lr != get_lr(optimizer):
        print("Loading best Model weights")
        model.load_state_dict(best_weights)
    
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_weights = copy.deepcopy(model.state_dict())
        print("Updating Best Model weights")
        
    loss_history["training_loss"].append(training_loss / train_len)
    loss_history["validation_loss"].append(validation_loss / val_len)
            
    print(f"\n{epoch + 1}/{epochs}")
    print(f"Training Loss: {training_loss / train_len}")
    print(f"Validation Loss: {validation_loss / val_len}")
    print("\n" + "*" * 50)

torch.save(best_weights, model_weights_file)
