#!/usr/bin/env python3

import sys
import os
from openvino.runtime import Core, PartialShape, serialize

def get_input_output_names(ports):
    return [port.any_name for port in ports]

def get_node_names(ports):
    return [port.node.friendly_name for port in ports]

def print_inputs_and_outputs_info(model):
    inputs = model.inputs
    input_names = get_input_output_names(inputs)
    for i in range(len(inputs)):
        print(f"Model input  {i:2}: {input_names[i]:20}: precision {inputs[i].element_type.get_type_name()}, "
              f"dimensions ({str(inputs[i].node.layout)}): "
              f"{' '.join(str(x) for x in inputs[i].partial_shape)}")
    outputs = model.outputs
    output_names = get_input_output_names(outputs)
    for i in range(len(outputs)):
        print(f"Model output {i:2}: {output_names[i]:20}: precision {outputs[i].element_type.get_type_name()}, "
              f"dimensions ({str(outputs[i].node.layout)}): "
              f"{' '.join(str(x) for x in  outputs[i].partial_shape)}")

def print_divider(label=None):
    dashed_line = '-'*100
    print(dashed_line)
    if label is not None:
        print(f"+ {label} " + "\n")

def reshape_ir(input_ir_path, output_ir_path):
    outdir = os.path.dirname(output_ir_path)
    if outdir != "":
        os.makedirs(os.path.dirname(output_ir_path), exist_ok=True)
    
    core = Core()
    ov_model = core.read_model(input_ir_path)

    print_divider("{}: Input Model Shape".format(os.path.basename(input_ir_path)))
    print_inputs_and_outputs_info(ov_model)

    new_iport_cfg = dict()
    for iport in ov_model.inputs:
        new_iport_cfg[iport.any_name] = PartialShape([1, 384])
    ov_model.reshape(new_iport_cfg)
    serialize(ov_model, output_ir_path, output_ir_path.replace(".xml", ".bin"))

    print_divider("{}: Output Model Shape".format(os.path.basename(output_ir_path)))
    print_inputs_and_outputs_info(ov_model)
    print_divider()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide path to model xml file as 1st arg and"
              " path to output xml as 2nd arg")
        exit()
    else:
        input_ir_path = sys.argv[1]
        output_ir_path = sys.argv[2]

        assert os.path.splitext(input_ir_path)[-1] != "xml", "Input IR must be in .xml file extension"
        assert os.path.splitext(output_ir_path)[-1] != "xml", "Output IR must be in .xml file extension"

        reshape_ir(input_ir_path, output_ir_path)
        print("Please find output IR at --> {}".format(output_ir_path))
        

