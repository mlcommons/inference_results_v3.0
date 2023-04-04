import os
import torch
import torchvision
import intel_extension_for_pytorch as ipex
import json
import torchvision.transforms as transforms
import torch.fx.experimental.optimization as optimization
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
from model_rn50_partitions import GC_RN50_Start, GC_RN50_End, GC_RN50_Middle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path","-m", required=True, help="Path to pth file")
    parser.add_argument("--data-path-cal", required=True, help="path to calibration dataset")
    parser.add_argument("--data-path-val", help="path to evaluation dataset")
    parser.add_argument("--accuracy", help="Check accuracy", action="store_true")
    parser.add_argument("--batch-size", "-b", help="Batch size", type=int, default=1)
    
    parser.add_argument('--model-int8-partition', help="Partition model into subsequent parts", action="store_true")
    parser.add_argument("--calibrate-full-weights", help="Generate weight scales and save fused fp weights", action='store_true')
    parser.add_argument("--save-full-weights", help="Save weights for backbone", action='store_true')
    parser.add_argument("--calibrate-start-partition", help="Generate quantized model for first part of rn50", action='store_true')
    parser.add_argument("--calibrate-end-partition", help="Generate quantized model for end part of rn50", action='store_true')
    parser.add_argument("--channels-last", help="Use channelsLast memory format", action='store_true')
    parser.add_argument("--save-dir", help="Where to save generated model files and weight scales", default="models")
    parser.add_argument("--massage", default=False, action="store_true", help="Remove quantization and dequantization of start portion")

    args = parser.parse_args()
    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def check_evaluate(model, args):

    if args.data_path_val is None:
        print("To run accuracy validation, path to validation dataset must be set")
        print("Run 'preprocess_imagenet_validation_data.py', then pass the path with --data_path_val")
        return

    if not os.path.isdir(args.data_path_val):
        print("Path to prepared validation data {} not found".format(args.data_path_val))
        print("Please double check")
        return

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data_path_val, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)

    with torch.no_grad():
        print("[MAIN] Checking accuracy with {} samples".format(len(val_loader)))
        for i, (images, target) in enumerate(val_loader):
            output = model(images)
            # measure accuracy and record losss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))      
   
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generateFullWeightsScale(rn50_model, cal_loader, args):
    scales_file = "/resnet50-int8-scales.json"

    # Fold bn into conv
    # model = rn50_model
    model = optimization.fuse(rn50_model, inplace=False)
    # print(model)
    
    # Save the fused model for weights to be passed for custom kernel
    if args.save_full_weights:
        with torch.no_grad():
            traced_script_module = torch.jit.trace(model, torch.randn(1,3,224,224),check_trace=False).eval()
        traced_script_module.save(args.save_dir + "/resnet50-full.pth")

    config = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.channels_last:
        x.to(memory_format=torch.channels_last)
    prepared_model = prepare(model, config, example_inputs=x, inplace=False)

    print("[MAIN] Starting quantization. Number available samples: {}".format(len(cal_loader)))
    for i, (images, target) in enumerate(cal_loader):     
        if i % 50 == 0:
            print(" Iteration step {}".format(i))

        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)
        prepared_model(images)
        

      
    prepared_model.save_qconf_summary(qconf_summary=args.save_dir + scales_file)
    print(".........calibration step done..........")
    print("[MAIN] INT8 scales saved to {}".format(scales_file))
   
    convert_model = convert(prepared_model)
    with torch.no_grad():
        if args.channels_last:
            traced_model = torch.jit.trace(convert_model, torch.randn(args.batch_size,3,224,224).to(memory_format=torch.channels_last))
        traced_model = torch.jit.freeze(traced_model)
    
    y = traced_model(torch.randn(args.batch_size, 3, 224, 224))
  
    output_model_name = args.save_dir + "/resnet50-int8-model.pth"
    traced_model.save(output_model_name)
    

    if args.accuracy:
        # Pass in the 50000 images
        print("[MAIN] Evaluating accuracy")

        # Reload saved model
        model = torch.jit.load(output_model_name, map_location="cpu")
        check_evaluate(model, args)


def generateStartModel(rn50_model, cal_loader, args):
    print("[MAIN] Quantizing start part")
    if os.path.exists('resnet50-int8-scales.json'):
        fr = open('resnet50-int8-scales.json', 'r')
        scales_full = json.load(fr)
        start_scale = float(scales_full[' ']['q_op_infos']['2']['input_tensor_infos'][0]['scale'][0])
        start_zero = int(scales_full[' ']['q_op_infos']['2']['input_tensor_infos'][0]['zero_point'][0])

        in_scale = float(scales_full[' ']['q_op_infos']['0']['input_tensor_infos'][0]['scale'][0])
        in_zero = int(scales_full[' ']['q_op_infos']['0']['input_tensor_infos'][0]['zero_point'][0])
    else:
        start_scale = float(1)
        start_zero = int(0)

        in_scale = float(1)
        in_zero = int(0)
    model = GC_RN50_Start(rn50_model, scale_out=start_scale, zero_out=start_zero, skip_quant=True, skip_quant_in=False, scale_in=in_scale, zero_in=in_zero)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    scales_file= args.save_dir + "/resnet50-start-int8-scales.json"
    config = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)

    prepared_model = ipex.quantization.prepare(model, config, example_inputs=x, inplace=False)
    print("[MAIN] Starting quantization. Number available samples: {}".format(len(cal_loader)))
    for i, (images, target) in enumerate(cal_loader):
        if i % 50 == 0:
            print(" Iteration step {}".format(i))
            
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        # with ipex.quantization.calibrate(conf):
        output = prepared_model(images)
    
    prepared_model.save_qconf_summary(qconf_summary=scales_file)
    convert_model = convert(prepared_model)
    # torch.save(convert_model, args.save_dir + '/resnet50-start-convert-model.pth')
    with torch.no_grad():
        traced_model = torch.jit.trace(convert_model, torch.randn(args.batch_size, 3, 224, 224))
        traced_model = torch.jit.freeze(traced_model)
        if args.massage:
            #========================Remove input quantization=========================
            # # Currently not possible. IPEX requires that dequant is after quant
            # # Fixed with a patch
            # Specific to the current ipex version. Other ipex versions won't work
            # Find the alias of the input node
            nodes_alias = traced_model.graph.findAllNodes("aten::alias")
            for node in nodes_alias:
                input_nodes = list(node.inputs())
                if len(input_nodes)>0:
                    if 'x' in input_nodes[0].debugName():
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
            # node_input_quant.removeAllInputs()
            #=======================Remove output dequantization========================
            # Find output of max pooling
            node_pool = traced_model.graph.findAllNodes("aten::max_pool2d")[0]
            node_pool_out = node_pool.output().debugName()
            # Find the alias of the max pooling output node
            nodes_alias = traced_model.graph.findAllNodes("aten::alias")
            for node in nodes_alias:
                input_nodes = list(node.inputs())
                if len(input_nodes)>0:
                    if input_nodes[0].debugName()==node_pool_out:
                        node_alias_out = node.output().debugName()
            # Find quantization that takes max pooling outputs
            nodes_quant = traced_model.graph.findAllNodes("aten::quantize_per_tensor")
            for node in nodes_quant:
                input_nodes = list(node.inputs())
                if len(input_nodes)>0:
                    if input_nodes[0].debugName()==node_alias_out:
                        output_pool_quant = node.output()
            traced_model.graph.eraseOutput(0)
            traced_model.graph.registerOutput(output_pool_quant)
    output_fn = args.save_dir + "/resnet50-start-int8-model.pth"
    traced_model.save(output_fn)

    return
     
   
def generateEndModel(rn50_model, cal_loader, args):

    print("[MAIN] Quantizing end part")

    # Scales for output quantization of the middle part
    if os.path.exists('resnet50-int8-scales.json'):
        fr = open('resnet50-int8-scales.json', 'r')
        scales_full = json.load(fr)
        end_scale = float(scales_full[' ']['q_op_infos']['119']['input_tensor_infos'][0]['scale'][0])
        end_zero = int(scales_full[' ']['q_op_infos']['119']['input_tensor_infos'][0]['zero_point'][0])
    else:
        end_scale = float(1)
        end_zero = int(0)

    # Requires start portion of model (conv->bn->relu->maxpool)
    start_model = GC_RN50_Start(rn50_model, skip_quant=True)
    start_model.eval()

    # Requires portions up to rn50_model.layer4
    mid_model = GC_RN50_Middle(rn50_model, scale_out=end_scale, zero_out=end_zero, skip_quant_in=True, skip_quant_out=False)
    mid_model.eval()

    # Create the end part starting from avgPool
    model = GC_RN50_End(rn50_model, scale_in=end_scale, zero_in=end_zero, skip_quant=False)
    model.eval()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    scales_file= args.save_dir + "/resnet50-end-int8-scales.json"
    config = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    x = torch.randn(args.batch_size, 2048, 7, 7).to(torch.int8)
    if args.channels_last:
        x.to(memory_format=torch.channels_last)
    prepared_model = ipex.quantization.prepare(model, config, example_inputs=x, inplace=False)

    print("[MAIN] Starting quantization. Number available samples: {}".format(len(cal_loader)))
    for i, (images, target) in enumerate(cal_loader):
        if i % 50 == 0:
            print(" Iteration step {}".format(i))

        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        start_out = start_model(images)
        mid_out = mid_model(start_out)

        # with ipex.quantization.calibrate(conf):
        output = prepared_model(mid_out)

    prepared_model.save_qconf_summary(qconf_summary=scales_file)

    convert_model = convert(prepared_model)
    with torch.no_grad():
        traced_model = torch.jit.trace(convert_model, x)
        traced_model = torch.jit.freeze(traced_model)

    output_fn = args.save_dir + "/resnet50-end-int8-model.pth"
    # torch.save(model)
    traced_model.save(output_fn)

    return

def main():


    args = parse_arguments()
    os.makedirs(args.save_dir, mode = 0o777, exist_ok=True)

    model = models.__dict__['resnet50'](pretrained=False)
    state_dict = torch.load(args.checkpoint_path,  map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    # Pass in the 500 images for calibration
    cal_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data_path_cal, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,

        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    
    # First generates the weights for the entire model (and save a conv+bn fused model
    # for custom kernel backbone
    if args.calibrate_full_weights:
        generateFullWeightsScale(model, cal_loader, args)

    
    # If generating int8 models for start and end portions of rn50
    if args.calibrate_start_partition:
        generateStartModel(model, cal_loader, args)

    # Quantize end portion of rn50: avgpool->fc
    if args.calibrate_end_partition:
        generateEndModel(model, cal_loader, args)


if __name__=="__main__":
    main()