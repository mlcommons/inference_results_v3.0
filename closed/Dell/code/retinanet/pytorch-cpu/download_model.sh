#!/bin/bash

wget https://zenodo.org/record/6605272/files/retinanet_model_10.zip?download=1 -O retinanet_model_10.zip
unzip retinanet_model_10.zip
mv retinanet_model_10.pth retinanet-model.pth
