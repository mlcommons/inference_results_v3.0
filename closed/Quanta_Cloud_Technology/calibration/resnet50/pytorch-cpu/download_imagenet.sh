#!/bin/bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

if [ -d "ILSVRC2012_img_val" ]; then
    rm -r ILSVRC2012_img_val
fi

mkdir ILSVRC2012_img_val

tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val

cp val_data/*.txt ILSVRC2012_img_val/
