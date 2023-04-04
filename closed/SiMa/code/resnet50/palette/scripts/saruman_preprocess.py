
import numpy as np
import cv2
# import av
import sys
import os
import csv
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image


"""
Pre-process Calibration Data 
"""



def center_crop(img, out_height, out_width):
    """Return a center crop of given size from input image."""
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5):
    """Use OpenCV to resize image."""
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def load_and_preproc_file(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = (224, 224)
    image = resize_with_aspectratio(image, h, w)
    image = center_crop(image, h, w)
    image = np.asarray(image, dtype='float32')
    # Normalize image.
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    image -= means
    # Transpose. to CHW
    image = image.transpose([2, 0, 1])
    # return image
    return image
    # return np.clip(image, -128.0, 127.0).astype(dtype=np.int8, order='C')


def walk_dir(path):
    l = []
    for root, dirs, files in os.walk(path):
        for name in files:
            l.append((name, os.path.join(root, name)))
    l = list(sorted(l, key=lambda x: x[0]))
    return l


def load_file_map(path):
    l = []
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            l.append((row[1], row[0]))
    l = sorted(l, key=lambda r: r[0])
    return np.array(l)


def load_all(dirname, cal_map_file=None):
    paths = walk_dir(dirname)
    imgs = []
    cal_map = None
    if cal_map_file is not None:
        with open(cal_map_file) as f:
            cal_map = f.read()
    for i in tqdm(range(len(paths))):
        filename, fullpath = paths[i]
        if cal_map is not None and filename not in cal_map:
            continue
        print(filename)
        img = load_and_preproc_file(fullpath)
        imgs.append(img)
    return np.array(imgs)


if __name__ == '__main__':
    example = '/media/data1/imagenet/ILSVRC2012/val/n07711569/ILSVRC2012_val_00000183.JPEG'
    img = load_and_preproc_file(example)
    print(img.shape)

    cal_map_file = 'path_to_cal_file/calibration/cal_map.txt'
    # save_to = './mlperf_resnet50_cal_/media/data1/imagenet/mlperf/mlperf_resnet50_dataset_CHW_LANCZOS4_normalized.dat'
    save_to = './mlperf_resnet50_cal_NCHW.dat'
    imgs = load_all('/media/data1/imagenet/ILSVRC2012/val/', cal_map_file=cal_map_file)
    print('Shape of images: ', imgs.shape)
    print('Spot Checks: imgs[0][0][0][0]', imgs[0][0][0][0])
    print('Spot Checks: imgs[1][0][0][0]', imgs[1][0][0][0])
    print('Spot Checks: imgs[0][1][0][0]', imgs[0][1][0][0])
    print('Spot Checks: imgs[0][0][1][0]', imgs[0][0][1][0])
    print('Spot Checks: imgs[0][0][0][1]', imgs[0][0][0][1])
    print('Spot Checks: imgs[2][2][2][2]', imgs[2][2][2][2])
    imgs.tofile(save_to)
    np.save(save_to, imgs)

