import pysc
import subprocess
import struct
import math
import argparse
from pysc.graph.configs import postop_type
from utils import *
import os.path

CUR_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
PROJ_DIR = os.path.split(CUR_DIR)[0]
default_outpath = os.path.join(PROJ_DIR, "src", "kernel_mlp")

parser = argparse.ArgumentParser()
parser.add_argument("--buildpath",
                    type=str, help="The cmake build path of graph-compiler")
parser.add_argument("--no-data-source", action="store_true", default=False,
                    help="Don't generate global data in C source code")
parser.add_argument("--extra-options", type=str,
                    default="", help="The extra Compiler options")
parser.add_argument("--compile", type=str, default="none",
                    help="Compile the kernel: none/bin/src")
parser.add_argument("--outpath", type=str, default=default_outpath,
                    help="The destination path to store the generated kernels")
args = parser.parse_args()


ctx = pysc.contexts.get_context("spr")  # pysc.get_default_context()
pysc.runtime.config.num_threads_per_instance = 56
out_path = args.outpath
if not os.path.exists(out_path):
    os.makedirs(out_path)


def get_fwd_graph(bs_str: str):
    test_layers = 3
    test_batch_size = batch_size_to_number(bs_str)
    test_hidden_size = [13, 512, 256, 128]
    postop_types = [postop_type.bias, postop_type.bias, postop_type.bias,
                    postop_type.relu, postop_type.relu, postop_type.relu]
    qinfo = []
    g = pysc.graph.predefined.get_mlp_training_forward_graph(
        [], test_layers, test_batch_size, test_hidden_size, pysc.dtype("bf16"), postop_types, [])
    g.attrs["temp.name"] = g.attrs["temp.name"].extract() + "_" + bs_str
    # print(g)
    for op in g:
        if op.op_name == "input" or op.op_name == "output":
            if get_op_name(op) == "input":
                op.attrs["keep_plain"] = True
            elif get_op_name(op) == "relu_out2":
                op.attrs["target_formats"] = make_vector_any(
                    "sc_data_format_t", [pysc.data_format.MK()])
    g.attrs["is_input_plain"] = False
    g.attrs["is_output_plain"] = False
    g.run_passes_and_tune(ctx, timeout=0)
    print(g)
    return g


def map_bwd_arg_to_fwd_arg(name: str) -> str:
    def process(f):
        return f if f != "input0" else "input"
    if name.startswith("out_grad_"):
        return process(name[len("out_grad_"):])
    ret = {"gradient": "relu_out2",
           "in_relu_output": "relu_out2",
           "data_input0": "input",
           "data_input1": "relu_out0",
           "data_input2": "relu_out1",
           }
    if name in ret:
        return ret[name]
    if name.startswith("weight_input"):
        return name.replace("_input", "")
    return None


def get_bwd_graph(bs_str: str, arg_name_to_format):
    test_layers = 3
    test_batch_size = batch_size_to_number(bs_str)
    test_hidden_size = [13, 512, 256, 128]
    postop_types = [postop_type.bias, postop_type.bias, postop_type.bias,
                    postop_type.relu, postop_type.relu, postop_type.relu]
    qinfo = []
    g = pysc.graph.predefined.get_mlp_training_backward_graph(
        [], test_layers, test_batch_size, test_hidden_size, pysc.dtype("bf16"), postop_types, [])
    g.attrs["temp.name"] = g.attrs["temp.name"].extract() + "_" + bs_str
    for op in g:
        if op.op_name == "output":
            detail = op.inputs[0].details
            the_format = arg_name_to_format[map_bwd_arg_to_fwd_arg(
                get_op_name(op))].format
            op.attrs["target_formats"] = make_vector_any(
                "sc_data_format_t", [the_format])
        if op.op_name == "input":
            detail = op.outputs[0].details
            detail.format = arg_name_to_format[map_bwd_arg_to_fwd_arg(
                get_op_name(op))].format
    print(g)
    g.attrs["is_input_plain"] = False
    g.attrs["is_output_plain"] = False
    g.run_passes_and_tune(ctx, timeout=0)
    print(g)
    return g


def gen_fwd(batch_size: str, reorders: dict):
    fwd_g = get_fwd_graph(batch_size)
    main_src, doc_src1, fwd_graph_name, reorder_to_add = compile_graph(fwd_g, ctx)
    out_data_file = os.path.join(out_path, "fwd_data" + batch_size + ".cpp")
    out_src_file = os.path.join(out_path, "fwd" + batch_size + ".cpp")
    (inited_size, uninited_size) = process_global_data(
        fwd_graph_name, out_data_file, args.no_data_source)
    decl = process_source(main_src, fwd_graph_name, inited_size,
                          uninited_size, out_src_file)
    doc_src = doc_src1 + decl
    for k in reorder_to_add:
        reorders[k] = reorder_to_add[k]
    fwd_arg2format = get_arg_name_format_map(fwd_g)
    return fwd_g, doc_src, fwd_arg2format


def gen_bwd(batch_size, fwd_g, fwd_arg2format, reorders):
    bwd_g = get_bwd_graph(batch_size, fwd_arg2format)
    bwd_arg2format = get_arg_name_format_map(bwd_g)
    main_src, doc_src1, bwd_graph_name, reorder_to_add2 = compile_graph(bwd_g, ctx)
    out_data_file = os.path.join(out_path, "bwd_data" + batch_size + ".cpp")
    out_src_file = os.path.join(out_path, "bwd" + batch_size + ".cpp")
    (inited_size, uninited_size) = process_global_data(
        bwd_graph_name, out_data_file, args.no_data_source)
    decl = process_source(main_src, bwd_graph_name, inited_size,
                          uninited_size, out_src_file)
    doc_src = doc_src1 + decl
    for k in reorder_to_add2:
        reorders[k] = reorder_to_add2[k]
    return bwd_g, doc_src, bwd_arg2format


reorder_to_add = dict()
doc_src =  '#pragma once\n#include <stdint.h>\n\n'

def gen_by_batch(bs_str):
    global doc_src
    global reorder_to_add
    fwd_g_4k, doc_src_tmp, fwd_arg2format_4k = gen_fwd(bs_str, reorder_to_add)
    doc_src += doc_src_tmp
    bwd_g_4k, doc_src_tmp, bwd_arg2format_4k = gen_bwd(
        bs_str, fwd_g_4k, fwd_arg2format_4k, reorder_to_add)
    doc_src += doc_src_tmp
    return (fwd_arg2format_4k, bwd_arg2format_4k)


def gen_by_batches(bs_list: list):
    return [(bs, gen_by_batch(bs)) for bs in bs_list]


bs2formats = gen_by_batches(["4k", "128k"])
for (bs, args2format) in bs2formats:
    print ("bs:", bs)
    print ("args2foramt:", args2format)

main_src, decl, reorder_info = make_reorder_graph(reorder_to_add, ctx)
reorder_data_file = os.path.join(out_path, "reorder_data.cpp")
(inited_size, uninited_size) = process_global_data("reorder", reorder_data_file,
                                                   args.no_data_source)
reorder_func_replace_map = []
for idx, info in enumerate(reorder_info):
    old_func_name = "reorder__{}(".format(idx+1)
    new_func_name = info[2]+"("
    reorder_func_replace_map.append((old_func_name, new_func_name))
reorder_func_replace_map.append(
    ('extern "C" void main_entry', 'static void main_entry'))
reorder_src_file = os.path.join(out_path, "reorder.cpp")
_ = process_source(main_src, "reorder", inited_size,
                   uninited_size, reorder_src_file, reorder_func_replace_map)
doc_src_reorder = '#pragma once\n#include <stdint.h>\n\n' + decl
with open(out_path+"/reorder.hpp", 'w') as f:
    f.write(doc_src_reorder)


with open(out_path+"/mlp.hpp", 'w') as f:
    f.write(doc_src)

shape_code_template = '''
namespace mlp_fwd_{bs}_shape {{
{fwd}
}}

namespace mlp_bwd_{bs}_shape {{
{bwd}
}}
'''

shape_src_template = '''#pragma once
{}
{}
'''

with open(out_path+"/shape.hpp", 'w') as f:
    code = "\n".join([shape_code_template.format(bs=bs, fwd=gen_doc_for_shape(fwd_arg2format), bwd=gen_doc_for_shape(
        bwd_arg2format)) for (bs, (fwd_arg2format, bwd_arg2format)) in bs2formats])
    f.write(shape_src_template.format("#include <array>", code))

for (bs, (fwd_arg2format, bwd_arg2format)) in bs2formats:
    del fwd_arg2format["input"]
    del fwd_arg2format["relu_out2"]

with open(out_path+"/pack.hpp", 'w') as f:
    code = "\n".join([shape_code_template.format(bs=bs, fwd=gen_src_for_pack_unpack(
        fwd_arg2format), bwd=gen_src_for_pack_unpack(bwd_arg2format)) for (bs, (fwd_arg2format, bwd_arg2format)) in bs2formats])
    f.write(shape_src_template.format('#include "reorder.hpp"\n#include <string.h>',code))
