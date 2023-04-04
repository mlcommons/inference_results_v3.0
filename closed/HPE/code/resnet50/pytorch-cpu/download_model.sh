#!/bin/bash
wget --no-check-certificate https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth -O resnet50.pth
mv resnet50.pth resnet50-fp32-model.pth
