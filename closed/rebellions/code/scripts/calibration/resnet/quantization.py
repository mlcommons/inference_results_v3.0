"""
This script is designed for quantize resnet to prepare MLPerf.

The goals are below:
1. Expand channel dim of the image tensor(4).

Usage : 
python3 preprocess.py \
    --calibration_method CALIBRATION_METHOD (minmax, l2) \
    --calibration_num_sample NUM \
    --pt output path of pytorch graph module (.pt)
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision

torch.manual_seed(0)
from torch.utils.data.dataloader import DataLoader

from rebel_algorithms.graph_view.torch_impl import TorchGraphView
from rebel_algorithms.algorithms.calibration import calibration, CALIBRATION_METHODS
from dataset import PyTorchImageFolder


@torch.no_grad()
def PTQ(
    graph: TorchGraphView,
    quant_bit=8,
    calibration_method=None,
    dataset=None,
    layer_from=None,
    layer_to=None,
):
    """
    params
        graph : TorchGraphView
        calibration_method : functions to calibrate weight and activations
            calibration_method(activations, quant_bit)
        dataset : iterable dataset object.
            (sample, label) = next(iter(dataset))
        model : the name of model.

    return
        TorchGraphView (with input quantizing node)
    """
    if quant_bit in [0, 16]:
        return graph

    assert quant_bit == 8
    assert calibration_method is not None
    assert dataset is not None

    # Select input node (Conv2d or Linear layer)
    layer_nodes = list(
        graph[lambda x: x.type == "Conv2d" or x.type == "Linear"][layer_from:layer_to]
    )

    layer_inputs = list()
    for layer_node in layer_nodes:
        layer_input = layer_node.inbounds[0]
        if layer_input not in layer_inputs:
            layer_inputs.append(layer_input)

    # Prepare batch input sample
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    sample, _ = next(iter(dataloader))  # batch, 4, 224, 224

    for layer_input in layer_inputs:
        outbound_nodes = [v for v in layer_input.outbounds]  # Cache it before refresh

        module = graph.generate(layer_input.inbounds[0])
        module.eval()

        print("Calibrating", layer_input.inbounds[0].name)

        activations = module(sample).numpy()
        activation_scale_factor = calibration(activations, 8, method=calibration_method)

        graph.quantize(layer_input, 8, activation_scale_factor)

        for outbound_node in outbound_nodes:
            if outbound_node.type == "Linear":
                weights = graph.get_parameter(outbound_node, "weight").data.numpy()
                weights_scale_factor = calibration(weights, 8, method="minmax")
                graph.quantize_parameter(outbound_node, "weight", 8, weights_scale_factor)
                graph.dequantize(
                    outbound_node.outbounds[0],
                    activation_scale_factor * weights_scale_factor,
                )

            elif outbound_node.type == "Conv2d":
                weights = graph.get_parameter(outbound_node, "weight").data.numpy()
                weights_scale_factor = calibration(
                    weights, 8, method="minmax", channelwise=True, channel_axis=0
                )
                graph.quantize_parameter(outbound_node, "weight", 8, weights_scale_factor)
                graph.dequantize(
                    outbound_node.outbounds[0],
                    activation_scale_factor * weights_scale_factor,
                )
            else:
                # Find target edge
                for outbound_parent_node in outbound_node.predecessors:
                    if outbound_parent_node.type == "to":
                        target_edge = outbound_parent_node.outbounds[0]
                        graph.dequantize(
                            target_edge,
                            activation_scale_factor,
                            [v for v in target_edge.outbounds if v is not outbound_node],
                        )
                        break
    return graph


def main(args):
    # No implementation for fp16/fp32
    assert args.quant_bit == 8

    with torch.no_grad():
        module = torchvision.models.resnet50(pretrained=False)
        module.load_state_dict(torch.load(args.pretrained_weight))
        module.eval()

    # Define a graph
    graph = TorchGraphView(module, None, [("x", torch.randn(1, 3, 224, 224, dtype=torch.float32))])

    with open(args.cal_image_list_option, "r") as f:
        cal_image_list = [line.strip() for line in f.readlines()]
        cal_image_list = [s.split(" ")[0] for s in cal_image_list]
        cal_image_list = [os.path.join(args.data_path, s) for s in cal_image_list]

    # Add quantization node on the graph
    dataset = PyTorchImageFolder(
        cal_image_list,
    )

    graph = PTQ(
        graph,
        quant_bit=args.quant_bit,
        calibration_method=args.calibration_method,
        dataset=dataset,
    )

    preprocessed_module = graph.generate().eval()
    torch.save(preprocessed_module, args.pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=Path,
        help="(Path) type, Path to the dataset.",
    )
    parser.add_argument(
        "--quant_bit",
        type=int,
        default=8,
        help="(int) type, quantization bits \
                Usage: --quant_bit=8",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        choices=list(CALIBRATION_METHODS),
        default="percentile",
        help="(str) type, \
            Calibration methods",
    )
    parser.add_argument(
        "--cal_image_list_option",
        default="cal_image_list_option_2.txt",
        type=str,
        help="calibration samples in dataset",
    )
    parser.add_argument(
        "--pt",
        type=Path,
        help="(Path) type, \
            Path to output pt graph file.",
    )
    parser.add_argument(
        "--pretrained_weight",
        type=Path,
        help="(Path) type, \
            pretrained resnet pth file.",
    )
    args = parser.parse_args()

    main(args)
