import json
import logging
import os
import sys
import time

import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
import torch.utils.data as data
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class OpenImages(data.Dataset):
    def __init__(self, data_path, annotations_file=None, transform=None, use_label_map=False):
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.data_path = data_path
        self.transform = transform
        self.use_label_map=use_label_map

        not_found = 0 
        empty_80catageories = 0
        if annotations_file is None:
            log.error("Path to annotations file required")
            sys.exit(1)

        self.annotation_file = annotations_file
        if not os.path.isdir(self.data_path):
            log.info("Data path not found: {}".format(self.data_path))
            sys.exit(1)

        if not os.path.exists(self.annotation_file):
            log.info("Annotations file not found: {}".format(self.annotation_file))
            sys.exit(1)

        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1

        start = time.time()
        images = {}
        with open(self.annotation_file, "r") as f:
            openimages = json.load(f)
        for i in openimages["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in openimages["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            image_name = img["file_name"]
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found += 1
                continue
            if len(img["category"])==0 and self.use_label_map: 
                #if an image doesn't have any of the 81 categories in it    
                empty_80catageories += 1 #should be 48 images - thus the validation sert has 4952 images
                continue 

            self.image_ids.append(image_id)
            self.image_list.append(img["file_name"]) #self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))


        self.label_list = np.array(self.label_list)

    
    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        htot, wtot = self.image_sizes[idx]
        fn = self.image_list[idx]
        #image_name = os.path.join("validation", "data", fn)
        src = os.path.join(self.data_path, fn)

        img = Image.open(src).convert("RGB")
        img = self.transform(img)
        return img, img_id, (htot, wtot), src

