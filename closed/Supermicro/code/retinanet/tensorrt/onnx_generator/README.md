# This folder contains scripts to generate the onnx and the anchors of RetinaNet.

## anchor_generator

The anchor_generator is modified and migrated from mlcommon training repo. The key difference
is that the original pytorch implementation generates anchors based on [800, 800] image size, 
in ltrb format. We need to scale is down to [1, 1] in xywh format in order to use with
EfficientNMS.

You can generate a set of anchors in different format by:
```
python3 -m code.retinanet.tensorrt.onnx_generator.anchor_generator
```

## patch_retinanet_efficientnms

The script uses the anchors, and adds the nms postprocessing by adding the EfficientNMS plugin
from TRT. The script also adds a concat plugin to concat all 4 outputs of EfficientNMS to match
mlperf requirements.

```
python3 patch_retinanet_efficientnms.py
``` 