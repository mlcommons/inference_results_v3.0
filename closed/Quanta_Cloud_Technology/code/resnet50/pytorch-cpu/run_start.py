import torch
import torch.nn.functional as F
import time
import argparse
# IPEX is needed for the code to run in quantized mode
import intel_extension_for_pytorch as ipex
import json

parser = argparse.ArgumentParser()
parser.add_argument("--start_path", type=str, default="models/resnet50-start-int8-model.pth", help="location of the start model")
parser.add_argument("--full_path", type=str, default="models/resnet50-full.pth", help="location of the full model")
parser.add_argument("--start_scales", type=str, default="models/resnet50-start-int8-scales.json", help='location of start quantization json')
parser.add_argument("--num_runs", type=int, default=10, help='Number of runs to do')
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--performance", default=False, action="store_true")

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # Load start portion of the model
    model = torch.jit.load(args.start_path)
    model.eval()
    params = list(torch.jit.load(args.full_path).named_parameters())
    for name,value in params:
        if name=='conv1.weight':
            conv_weight = value
        if name=='conv1.bias':
            conv_bias = value
    fr = open(args.start_scales, 'r')
    scales = json.load(fr)
    input_scale = scales[' ']['q_op_infos']['0']['input_tensor_infos'][0]['scale'][0]
    weight_scale = torch.tensor(scales[' ']['q_op_infos']['0']['weight_tensor_infos'][0]['scale'])
    output_scale = 0.05720944702625275
    fr.close()


    if args.performance:
        # Run the start model
        # Channel-last format, kernel tuned for batch size=9
        input_rand = torch.randn(2, 3, 224, 224).to(dtype=torch.float32, memory_format=torch.channels_last)
        with torch.no_grad():
            end = time.time()
            for i in range(args.num_runs):
                output_rand = model.forward(input_rand)
            print("Total time spent on {0} runs is {1} seconds".format(args.num_runs, time.time()-end))
    else:
        input_rand = torch.randn(1, 3, 224, 224).to(dtype=torch.float32, memory_format=torch.channels_last)
        input_quant = torch.quantize_per_tensor(input_rand, 0.02070588245987892, 0, torch.qint8)#.int_repr()
        weight_quant = torch.quantize_per_channel(conv_weight, weight_scale, torch.zeros_like(weight_scale).int(), 0, torch.qint8)#.int_repr()
        # bias_quant = torch.quantize_per_channel(conv_bias, weight_scale, torch.zeros_like(weight_scale), 0, torch.qint8)
        with torch.no_grad():
            output_correct = model.forward(input_quant.int_repr()).int_repr()
            print(output_correct)
            output_norm = F.conv2d(input_quant.int_repr().float(), weight_quant.int_repr().float(), None, 2, 3)
            output_norm = output_norm * (weight_scale*input_scale).reshape(1,-1,1,1)
            
            output_norm = output_norm + conv_bias.reshape(1,-1,1,1)
            output_norm = F.relu(output_norm)
            # print(output_norm[0,0,:10,:10])

            output_norm = F.max_pool2d(output_norm, 3, 2, 1)
            # Output quantization
            output_norm = torch.quantize_per_tensor(output_norm, output_scale, 0, torch.qint8).int_repr()
            print(output_norm)

if __name__=="__main__":
    main()