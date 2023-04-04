import torch
import cv2
import io
from PIL import Image
from torchvision import datasets, models

import torchvision.transforms as T

import os
import argparse

preprocess = T.Compose([
   T.Resize(256),
   T.CenterCrop(224),
   T.ToTensor(),
   T.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])

# Reads every image from text file, preprocesses and stores as a tensor .

def process_images(text_filename, output_dir, output_basename_tensor):
    if not os.path.exists(os.path.join(output_dir,text_filename)):
        print("Could not locate {}".format(text_filename))

    
    save_path = os.path.join(output_dir, output_basename_tensor)

    file_open = open(os.path.join(output_dir,text_filename), 'r')
    image_name_list = file_open.readlines()
    num_images = len(image_name_list)
    preprocessed_tensor = torch.zeros(num_images,3,224,224)

    for i in range(num_images):
        img = Image.open(image_name_list[i][:-1])
        img = img.convert('RGB')
        img = preprocess(img)
        preprocessed_tensor[i] = img
    
    f = io.BytesIO()
    torch.save(preprocessed_tensor, f, _use_new_zipfile_serialization=True)
    with open(save_path, 'wb') as fid:
        fid.write(f.getbuffer())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_filename", required=True, help="Full path to image to process")
    parser.add_argument("--output-dir", help="Save directory for processed image", default="preprocessed_data")
    parser.add_argument("--output_basename_tensor", help="basename for processed image")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, mode=0o777, exist_ok=True)
    process_images(args.text_filename, args.output_dir, args.output_basename_tensor)
    

if __name__=="__main__":
    main()