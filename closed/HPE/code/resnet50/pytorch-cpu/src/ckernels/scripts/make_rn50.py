import json
import argparse
import os.path
import os

import pysc
from pysc.graph.configs import quantize_infos

import graph_configs
from utils import *

reorder_to_add = dict()
doc_src = '#pragma once\n#include <stdint.h>\n\n'


def main():
    args = parse_args()
    ctx = pysc.contexts.get_context("spr")  # pysc.get_default_context()

    out_path = args.outpath
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    bs_args2formats = gen_by_batches_rn50(ctx, args, ["256", "8", "4"])

    # debug info
    for (bs, args2format) in bs_args2formats:
        print("bs:", bs)
        print("args2foramt:", args2format)
    print("gen_rn50_backbone done...\n")

    main_src, decl, reorder_info = make_reorder_graph(reorder_to_add, ctx)
    reorder_data_file = os.path.join(out_path, "reorder_data.cpp")
    (inited_size, uninited_size) = process_global_data("reorder",
                                                       reorder_data_file,
                                                       args.no_data_source)
    reorder_func_replace_map = []
    for idx, info in enumerate(reorder_info):
        old_func_name = "reorder__{}(".format(idx + 1)
        new_func_name = info[2] + "("
        reorder_func_replace_map.append((old_func_name, new_func_name))
    reorder_func_replace_map.append(
        ('extern "C" void main_entry', 'static void main_entry'))
    reorder_src_file = os.path.join(out_path, "reorder.cpp")
    _ = process_source(main_src, "reorder", inited_size, uninited_size,
                       reorder_src_file, reorder_func_replace_map)
    doc_src_reorder = '#pragma once\n#include <stdint.h>\n\n' + decl

    reorder_header_file = os.path.join(out_path, "reorder.hpp")
    with open(reorder_header_file, 'w') as f:
        f.write(doc_src_reorder)

    api_header_file = os.path.join(out_path, "rn50_backbone.hpp")
    with open(api_header_file, 'w') as f:
        f.write(doc_src)

    shape_code_template = '''
namespace backbone_{bs}_shape {{
{shapes}
}}
'''

    shape_src_template = '''#pragma once
{}
{}
'''

    shape_header_file = os.path.join(out_path, "shape.hpp")
    with open(shape_header_file, 'w') as f:
        code = "\n".join([
            shape_code_template.format(bs=bs,
                                       shapes=gen_doc_for_shape(arg2format))
            for (bs, arg2format) in bs_args2formats
        ])
        f.write(shape_src_template.format("#include <array>", code))

    pack_header_file = os.path.join(out_path, "pack.hpp")
    with open(pack_header_file, 'w') as f:
        code = "\n".join([
            shape_code_template.format(
                bs=bs, shapes=gen_src_for_pack_unpack(arg2format))
            for (bs, arg2format) in bs_args2formats
        ])
        f.write(
            shape_src_template.format(
                '#include "reorder.hpp"\n#include <string.h>', code))


def parse_args():
    CUR_DIR = os.path.dirname(
        os.path.abspath(__file__))  # This is your Project Root
    PROJ_DIR = os.path.split(CUR_DIR)[0]
    default_outpath = os.path.join(PROJ_DIR, "src", "kernel_rn50")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buildpath", type=str,
        help="The cmake build path of graph-compiler")  # unused
    parser.add_argument("--no-data-source",
                        action="store_true",
                        default=False,
                        help="Don't generate global data in C source code")
    parser.add_argument("--extra-options",
                        type=str,
                        default="",
                        help="The extra Compiler options")  # unused
    parser.add_argument("--compile",
                        type=str,
                        default="none",
                        help="Compile the kernel: none/bin/src")  # unused
    parser.add_argument(
        "--outpath",
        type=str,
        default=default_outpath,
        help="The destination path to store the generated kernels")
    args = parser.parse_args()
    return args


def get_qinfos(path="./int8-scales-fused.json"):
    with open(path, "r") as f:
        data = json.load(f)
        return get_qinfos_v1(path) if isinstance(data,
                                                 list) else get_qinfos_v2(path)


def get_qinfos_v1(path):
    print("read qinfo from ", path)
    qinfos = [() for _ in range(100)]

    with open(path, "r") as f:
        data = json.load(f)
        cnt = 0
        layer_start = [3, 25, 54, 97, 119]
        layer_index = 0
        index = 0
        old_index = 0
        conv_index = 0
        for i, item in enumerate(data):

            if item['id'] < 3 or item['id'] > 118:
                continue

            # print(item["id"], item["name"], item["inputs_flow"], item["outputs_flow"] )

            if item["id"] >= layer_start[layer_index]:
                old_index = index
                layer_index += 1
                index += 1
                conv_index = 0

            # quantize_infos_t(dtype, scales,
            #                 zero_points, per_channel, channel_axis,
            #                 asymmetric, dynamic)

            if item["name"] == "conv2d":

                conv_index += 1
                q1 = quantize_infos(pysc.dtype("s8"), item["input_scales"],
                                    [0], False, 0, False, False)

                q2 = quantize_infos(pysc.dtype("s8"), item["weight_scales"][0],
                                    [0], True, 0, False, False)
                cnt += 1
                if conv_index == 4:
                    qinfos[old_index] = (q1, q2)
                else:
                    qinfos[index] = (q1, q2)
                    index += 1

            elif item["name"] == "add":
                q1 = quantize_infos(pysc.dtype("s8"),
                                    [item["input_scales"][1]], [0], False, 0,
                                    False, False)
                q2 = quantize_infos(pysc.dtype("s8"),
                                    [item["input_scales"][0]], [0], False, 0,
                                    False, False)

                # conv block add
                if item['id'] in (9, 31, 60, 103):
                    q1 = quantize_infos(pysc.dtype("s8"),
                                        [item["input_scales"][1]], [0], False,
                                        0, False, False)

                q3 = quantize_infos(pysc.dtype("s8"),
                                    data[i + 2]["input_scales"], [0], False, 0,
                                    False, False)

                qinfos[index] = (q1, q1)
                index += 1
                qinfos[index] = (q2, q2)
                index += 1
                qinfos[index] = (q3, q3)
                index += 1

            # if item["id"] in [24, 53, 96, 118]:
            #     print("------------")
    return qinfos


def get_qinfos_v2(path):
    print("read qinfo from ", path)
    qinfos = [() for _ in range(100)]

    with open(path, "r") as f:
        data = json.load(f)[" "]["q_op_infos"]
        cnt = 0
        layer_start = [3, 25, 54, 97, 119]
        layer_index = 0
        index = 0
        old_index = 0
        conv_index = 0
        for i in range(122):
            item = data[str(i)]
            # print(item["op_type"])

            if i < 3 or i > 118:
                continue

            # print(item["id"], item["name"], item["inputs_flow"], item["outputs_flow"] )

            if i >= layer_start[layer_index]:
                old_index = index
                layer_index += 1
                index += 1
                conv_index = 0

            # quantize_infos_t(dtype, scales,
            #                 zero_points, per_channel, channel_axis,
            #                 asymmetric, dynamic)

            if item["op_type"] == "<class 'torch.nn.modules.conv.Conv2d'>":

                conv_index += 1
                input_tensor_infos = item["input_tensor_infos"][0]
                q1 = quantize_infos(pysc.dtype("s8"),
                                    input_tensor_infos["scale"], [0], False, 0,
                                    False, False)

                weight_tensor_infos = item["weight_tensor_infos"][0]
                q2 = quantize_infos(pysc.dtype("s8"),
                                    weight_tensor_infos["scale"], [0], True, 0,
                                    False, False)
                cnt += 1
                if conv_index == 4:
                    qinfos[old_index] = (q1, q2)
                else:
                    qinfos[index] = (q1, q2)
                    index += 1

            elif item[
                    "op_type"] == "<method 'add' of 'torch._C._TensorBase' objects>":

                residual_input = item["input_tensor_infos"][1]
                main_branch_input = item["input_tensor_infos"][0]

                q1 = quantize_infos(pysc.dtype("s8"), residual_input["scale"],
                                    [0], False, 0, False, False)

                q2 = quantize_infos(pysc.dtype("s8"),
                                    main_branch_input["scale"], [0], False, 0,
                                    False, False)

                # conv block add
                if i in (9, 31, 60, 103):
                    q1 = quantize_infos(pysc.dtype("s8"),
                                        residual_input["scale"], [0], False, 0,
                                        False, False)

                next_input = data[str(i + 2)]["input_tensor_infos"][0]
                q3 = quantize_infos(pysc.dtype("s8"), next_input["scale"], [0],
                                    False, 0, False, False)

                qinfos[index] = (q1, q1)
                index += 1
                qinfos[index] = (q2, q2)
                index += 1
                qinfos[index] = (q3, q3)
                index += 1

            # if item["id"] in [24, 53, 96, 118]:
            #     print("------------")

    return qinfos


def get_rn50_backbone(ctx, batch_size: str):
    # args, batch_size, is_quantize, add_type(0: f32, 1: qadd, 2: int8_add), cfgs, qinfos

    if batch_size not in ["256", "8", "4"]:
         raise ValueError('Unsupported batch size {}'.format(batch_size))
    get_cfgs = graph_configs.get_unified_cfgs

    cfgs = []
    for cfg in get_cfgs():
        cfgs.append(pysc.graph.configs.conv_fwd_config(**cfg.to_dict()))
    g = pysc.graph.predefined.get_rn50_backbone_graph(
        [],
        int(batch_size),
        True,
        pysc.graph.configs.Residual_Add_Type(2),
        cfgs,
        get_qinfos("./int8-scales-fused.json"),
        data_f32=False,
        output_f32=False)
    g.attrs["temp.name"] = "rn50_backbone_bs" + batch_size
    for op in g:
        if op.op_name == "input" or op.op_name == "output":
            if get_op_name(op) == "input":
                op.attrs["keep_plain"] = True
    g.attrs["is_input_plain"] = True
    g.attrs["is_output_plain"] = False
    g.run_passes_and_tune(ctx, timeout=0)
    return g


def gen_rn50_backbone(ctx, args, batch_size: str, reorders: dict):
    g = get_rn50_backbone(ctx, batch_size)
    print(g)
    main_src, doc_src1, graph_name, reorder_to_add = compile_graph(g, ctx)

    out_data_file = os.path.join(args.outpath,
                                 "backbone_data_" + batch_size + ".cpp")
    out_src_file = os.path.join(args.outpath,
                                "backbone_" + batch_size + ".cpp")
    (inited_size, uninited_size) = process_global_data(graph_name,
                                                       out_data_file,
                                                       args.no_data_source)
    decl = process_source(main_src, graph_name, inited_size, uninited_size,
                          out_src_file)
    doc_src = doc_src1 + decl
    for k in reorder_to_add:
        reorders[k] = reorder_to_add[k]
    arg2format = get_arg_name_format_map(g)
    return g, doc_src, arg2format


def gen_by_batch_rn50(ctx, args, batch_size: str):
    global doc_src
    global reorder_to_add

    if int(batch_size) == 4:
        ## enable multi-core for parallel for
        pysc.runtime.config.num_threads_per_instance = 2
        os.environ["IMAGE_AFFINITY_BOUNDARY"] = "12"
    else:
        ## use single-core w/o parallel for
        pysc.runtime.config.num_threads_per_instance = 1
        os.environ["IMAGE_AFFINITY_BOUNDARY"] = "12"

    g, doc_src_tmp, args2format = gen_rn50_backbone(ctx, args, batch_size,
                                                    reorder_to_add)
    doc_src += doc_src_tmp
    del os.environ["IMAGE_AFFINITY_BOUNDARY"]

    return args2format


def gen_by_batches_rn50(ctx, args, bs_list: list):
    return [(bs, gen_by_batch_rn50(ctx, args, bs)) for bs in bs_list]


if __name__ == "__main__":
    main()