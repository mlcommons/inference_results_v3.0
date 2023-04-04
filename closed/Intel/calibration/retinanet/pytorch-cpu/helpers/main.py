import argparse
import time
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.fx.experimental.optimization as optimization
from torch.utils.data import DataLoader
import numpy as np
import cv2
import json

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, HistogramObserver, MovingAverageMinMaxObserver

from model.retinanet import RetinaNet, retinanet_resnext50_32x4d_fpn
from model.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from model.feature_pyramid_network import LastLevelP6P7
from model.openimages import OpenImages

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path","-m", required=True, help="Path to pth file")
    parser.add_argument("--data-path", required=True, help="path to dataset")
    parser.add_argument("--annotation-file", help="path to annotation")
    parser.add_argument("--precision", help="Precision", choices=("int8", "fp32"), default="fp32")
    parser.add_argument("--quantized-weights", help="INT8 quantized scales. Required if --precision=int8", type=str, default="")
    parser.add_argument("--num-classes", help="Number of classes for the trained model", type=int, default=91)
    parser.add_argument("--num-iters", help="Number of iterations to run", type=int, default=100)
    parser.add_argument("--accuracy", help="Check accuracy", action="store_true")
    parser.add_argument("--batch-size", "-b", help="Batch size", type=int, default=1)
    parser.add_argument("--enable-fusion", help="Fuse layers", action="store_true")
    parser.add_argument("--load-state-dict", help="Set flag to call 'load_state_dict'", action='store_true')
    parser.add_argument("--save-trace-model", help="Whether to save the model as scriptmodule (and return", action="store_true")
    parser.add_argument("--save-trace-model-path", help="Path to save the scriptmodule (if --save-trace-model", type=str)
    parser.add_argument("--save-random-weights", help="Save model with random weights", action='store_true')
    parser.add_argument("--save-random-weights-path", help="Path to save random weights")
    parser.add_argument("--traced-model", help="Whether the checkpoint is traced and can be used for inference right away (like most fp32 pth files)", action="store_true")
    parser.add_argument("--calibrate", help="Generate quantization scales and return", action='store_true')
    parser.add_argument("--scales-output-file", help="Output path for quantization scales (if --calibrate)")
    parser.add_argument("--cal-iters", help="Number of calibration iterations", type=int, default=500)
    parser.add_argument("--warmups", help="Number of warmup", type=int, default=0)

    args = parser.parse_args()
    return args

def run_test(model, data_loader, num_iters, batch_size, acc_check, num_classes, cocoGt, args):

    if acc_check:
        model_acc = retinanet_resnext50_32x4d_fpn(num_classes=num_classes, pretrained=None, image_size=[800, 800])

    infer_time = 0
    decode_time = 0
    ret = []
    image_ids = []
    lines = []
    width, height = 800, 800
    MS = 1000
    num_iters = num_iters // args.batch_size # Scale Number of iterations to run by batch size
    print("[RUNTEST] Starting warmup")
    for i in range(args.warmups):
        x = torch.randn(args.batch_size, 3, 800, 800).to(memory_format=torch.channels_last)
        res = model(x)

    print("[RUNTEST] Warmup completed")

    print("[RUNTEST] Running test (available samples = {})".format(len(data_loader)))
    with torch.no_grad():
        for nbatch, (img, img_id, img_size, img_path) in enumerate(data_loader):
            img = img.to(memory_format=torch.channels_last)

            if nbatch==num_iters:
                break

            img = img.to(torch.device("cpu"))
            t0 = time.time()
            results = model(img)
            t1 = time.time()
            infer_time += (t1 - t0) * MS
            
            if not acc_check:
                if (nbatch % 10 ==0):
                    print("Inference: {:.3f}".format((t1-t0)*MS))
                continue
            
            p0 = time.time()
            results = model_acc.postprocess_detections(results)
            p1 = time.time()
            decode_time += (p1 - p0) * MS

            if (nbatch % 10 ==0):
                print("Inference: {:.3f};\tDecode: {:.3f}".format((t1-t0)*MS, (p1-p0)*MS ))

            for img_id_, img_size_, img_path_, result in zip(img_id, img_size, img_path, results):
                htot_, wtot_ = img_size_
                loc = result['boxes'].cpu().numpy()
                label = result['labels'].cpu().numpy()
                prob = result['scores'].cpu().numpy()

                line = "\n" + "="*10 + " {} ".format(img_id_) + "=" * 10 + "\n"
                lines.append(line)
                for loc_, label_, prob_ in zip(loc, label, prob):

                    xmin = loc_[0] / width
                    ymin = loc_[1] / height
                    xmax = loc_[2] / width
                    ymax = loc_[3] / height

                    boxes = [xmin*wtot_, ymin*htot_, (xmax-xmin)*wtot_, (ymax-ymin)*htot_]
                    detection = {"image_id": img_id_,
                            "image_loc": img_path_,
                            "category_id": int(label_),
                            "bbox": boxes,
                            "score": float(prob_),}

                    ret.append(detection)

                    line = ", ".join([str(el) for el in detection]) 
                    lines.append(line)

                    image_ids.append(img_id_)

    print("=====================================\n"*2)
    avg_infer_time = infer_time / (num_iters * args.batch_size)
    print("Mean inference time: {:.3f}".format(args.batch_size * avg_infer_time))
    avg_decode_time = 0
    if acc_check:
        avg_decode_time = decode_time / (num_iters * args.batch_size)
        print("Average decode time: {:.3f}".format(args.batch_size * avg_decode_time))

    avg_latency = avg_infer_time + avg_decode_time
    print("Avg latency: {:.3f}".format(args.batch_size * avg_latency))
    print("Throughput: {:.3f}".format(1000 / (avg_latency)))
    print("=====================================\n"*2)


    if not acc_check or len(ret)==0:
        return

    detections_json = "Detections-{}.json".format(num_classes)
    with open(detections_json, "w") as fp:
        json.dump(ret, fp, sort_keys=True, indent=4)

    openImagesDt = cocoGt.loadRes(detections_json)
    E = COCOeval(cocoGt, openImagesDt, iouType='bbox')
    E.params.imgIds = image_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    threshold = 0.376
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    print("Accuracy: {:.5f} ".format(E.stats[0]))

def collate_fn(data_list):
    imgs = []
    ids = []
    img_sizes = []
    img_names = []
    for tup in data_list:
        imgs.append(tup[0])
        ids.append(tup[1])
        img_sizes.append(tup[2])
        img_names.append(tup[3])

    return torch.stack(imgs), tuple(ids), tuple(img_sizes), tuple(img_names)


def main():

    args = parse_arguments()

    print("\n")
    print(args)
    print("\n")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    trans_val = transforms.Compose([
        transforms.Resize((800,800)),
        transforms.ToTensor(),
        normalize
        ])

    annotation_file = args.annotation_file
    if not annotation_file:
        if not os.path.exists(annotation_file):
            print("Please provide path to annotations file")
            sys.exit(1)
    else:
        if not os.path.exists(annotation_file):
            print("Provided annotations file not found: {}".format(annotation_file))
            sys.exit(1)


    openImagesGt = COCO(annotation_file=annotation_file)
    val_dataset = OpenImages(args.data_path, annotations_file=annotation_file, transform=trans_val)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False,
                            sampler=None,
                            num_workers=0,
                            collate_fn=collate_fn)

    if len(val_dataloader)==0:
        print("[ERROR] Calibration requires at least 1 calibration sample/image")
        print("[INFO ] Check README for instruction on how to set calibration data and annotations")
        sys.exit(1)

    print("Available samples {}".format(len(val_dataloader)))
    model = retinanet_resnext50_32x4d_fpn(num_classes=args.num_classes, pretrained=None, image_size=[800, 800])

    # If we're not using a saved traced model
    if not args.traced_model:
        od = torch.jit.load(args.checkpoint_path) #, map_location="cpu")
        model.load_state_dict(od.state_dict())
        model.eval()
        model.to(memory_format=torch.channels_last)

        model = optimization.fuse(model, inplace=False)

    

        if args.calibrate:
            qconfig = QConfig(activation=HistogramObserver.with_args(qscheme=torch.per_channel_symmetric, dtype=torch.qint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)) # Better acc=54.092, fps=265
        
            scales_file = args.scales_output_file
            if not scales_file:
                scales_file = "int8-scales-{}.json".format(args.num_classes)

            example_inputs = torch.randn(args.batch_size,3,800,800)
            example_inputs.to(memory_format=torch.channels_last)
            model = prepare(model, qconfig, example_inputs=example_inputs, inplace=False)
            print("[MAIN] Starting quantization. Number available samples: {}".format(len(val_dataloader)))

            for nbatch, (img, img_id, img_size, img_name) in enumerate(val_dataloader):
                img = img.to(memory_format=torch.channels_last)
                res = model(img)
                if nbatch % 10 == 0:
                    print("Step {}".format(nbatch))

                if nbatch==args.cal_iters:
                    break

            model.save_qconf_summary(qconf_summary=scales_file)
            print("[MAIN] Quantization parameters saved to {}".format(scales_file))

            # Save the model
            converted_model = convert(model)
            with torch.no_grad():
                traced_model = torch.jit.trace(converted_model, torch.randn(args.batch_size,3,800,800).to(memory_format=torch.channels_last))
                traced_model = torch.jit.freeze(traced_model)

            #========================Remove input quantization=========================
            # # Currently not possible. IPEX requires that dequant is after quant
            # # Fixed with a patch
            # Specific to the current ipex version. Other ipex versions won't work
            # Find the alias of the input node
            nodes_alias = traced_model.graph.findAllNodes("aten::alias")
            for node in nodes_alias:
                input_nodes = list(node.inputs())
                if len(input_nodes)>0:
                    if 'images' in input_nodes[0].debugName():
                        input_node_raw = input_nodes[0]
                        node_alias_input = node.output().debugName()
            # Find output node of input quantization
            nodes_quant = traced_model.graph.findAllNodes("aten::quantize_per_tensor")
            for node in nodes_quant:
                input_nodes = list(node.inputs())
                for input_node in input_nodes:
                    if input_node.debugName()==node_alias_input:
                        node_input_quant = node.output().debugName()
            # Find the alias of the input quantization output node
            nodes_alias = traced_model.graph.findAllNodes("aten::alias")
            for node in nodes_alias:
                input_nodes = list(node.inputs())
                if len(input_nodes)>0:
                    if input_nodes[0].debugName()==node_input_quant:
                        node_alias_quant = node.output().debugName()
                        node_alias = node
            # Find the _make_per_tensor_quantized_tensor node
            nodes_make = traced_model.graph.findAllNodes("aten::_make_per_tensor_quantized_tensor")
            for node in nodes_make:
                input_nodes = list(node.inputs())
                input_nodes[0] = input_node_raw
                node.removeAllInputs()
                for n in input_nodes:
                    node.addInput(n)
                output_make = node.output()
                node.moveBefore(node_alias)
            # Find dequantization that takes input quantization
            # Replace the input with the output of _make_per_tensor_quantized_tensor
            nodes_dequant = traced_model.graph.findAllNodes("aten::dequantize")
            for node in nodes_dequant:
                input_nodes = list(node.inputs())
                if input_nodes[0].debugName()==node_alias_quant:
                    input_nodes[0] = output_make
                node.removeAllInputs()
                for n in input_nodes:
                    node.addInput(n)

            if not args.save_trace_model_path:
                output_fn = "retinanet-{}-{}".format(args.precision, args.num_classes)
                output_fn += "-traced-model.pth"

            else:
                output_fn = args.save_trace_model_path

            traced_model.save(output_fn)
            print("[MAIN] Traced model saved at {}".format(output_fn))

            return

        elif args.precision=="int8":
            if not args.quantized_weights:
                print("INT8 requires quantization scales (a json file)")
                sys.exit(1)

            if not os.path.exists(args.quantized_weights):
                print("Cannot locate INT8 scales file {}".format(args.quantized_weights))
                sys.exit(1)

            print("[MAIN] Loading quantized scales")

            qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

            example_inputs = torch.randn(args.batch_size,3,800,800)
            example_inputs.to(memory_format=torch.channels_last)
            t0 = time.time()
            model = prepare(model, qconfig, example_inputs=example_inputs, inplace=False)
            model.load_qconf_summary(qconf_summary=args.quantized_weights)
            elapsed = (time.time() - t0) * 1000
            print("[MAIN] Time taken to load quantized weights: {} ms".format(round(elapsed,2)))
            model = convert(model)
            with torch.no_grad():
                model = torch.jit.trace(model, example_inputs)
                model = torch.jit.freeze(model)

            y = model(torch.randn(args.batch_size, 3, 800, 800))

    else:
        model = torch.jit.load(args.checkpoint_path, map_location="cpu")

    print("\n")
    print("************* Running Inference *********************")
    if args.traced_model:
        print("[MAIN] Using Traced model")
    print("\n")

    run_test(model, val_dataloader, args.num_iters, args.batch_size, args.accuracy, args.num_classes, openImagesGt, args)


if __name__=="__main__":
    main()
