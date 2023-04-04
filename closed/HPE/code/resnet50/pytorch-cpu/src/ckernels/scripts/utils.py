import pysc
import subprocess
import struct
import math
import os.path
import re

def run_cmd_get_output(cmd, cwd: str = None) -> str:
    p1 = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, cwd=cwd)
    out, err = p1.communicate()
    if p1.wait() != 0:
        print("Running command", cmd, "failed")
        exit(127)
    out = out.decode("utf-8")
    return out


def run_cmd(cmd: str, cwd: str, allow_failure=False):
    process = subprocess.Popen(cmd, cwd=cwd)
    ret = process.wait()
    if ret != 0:
        print("Running command", cmd, "failed")
        if not allow_failure:
            exit(127)
    return ret


def dims_to_str(dims: list):
    return 'x'.join([str(v) for v in dims])


def get_op_name(op):
    ## hack for rn50
    # if not "temp.name" in op.attrs:
    #     return None
    return op.attrs["temp.name"].extract()


def make_vector_any(typename, lst) -> pysc.any:
    any_str = [str(pysc.any(f)) for f in lst]
    data = '{{"type":"v[{typename}]", "data":[{data}]}}'.format(
        typename=typename, data=','.join(any_str))
    return pysc.any.make(data)


def sort_dict(v: dict):
    ret = [kv for kv in v.items()]
    ret.sort(key=lambda kv: kv[0])
    return ret


def batch_size_to_number(bs: str) -> int:
    if bs.endswith('k'):
        return int(bs[:-1])*1024
    return int(bs)


def get_arg_name_format_map(g: pysc.graph):
    ret = dict()
    for op in g:
        # print("op.name:" , op.op_name)
        # print (op.attrs)

        if op.op_name == "output":
            detail = op.inputs[0].details
            ret[get_op_name(op)] = detail
        if op.op_name == "input":
            detail = op.outputs[0].details
            ret[get_op_name(op)] = detail
    return ret


def gen_doc_for_shape(arg_name_to_format) -> str:
    templ = '''inline const std::array<int, {N}>& {name}() {{
    static const std::array<int, {N}> shape = {{{values}}};
    return shape;
}}
'''
    lines = []
    for name, fmt in sort_dict(arg_name_to_format):
        real_dims = fmt.real_dims
        line = templ.format(N=len(real_dims), name=name,
                            values=','.join([str(v) for v in real_dims]))
        lines.append(line)
    return "\n".join(lines)


def gen_src_for_pack_unpack(arg_name_to_format) -> str:
    plain_templ = '''inline void pack_{name}(uint16_t* out, uint16_t* in) {{
    memcpy(out, in, sizeof(uint16_t) * {num_elem});
}}

inline void unpack_{name}(uint16_t* out, uint16_t* in) {{
    memcpy(out, in, sizeof(uint16_t) * {num_elem});
}}
'''

    block_templ = '''inline void pack_{name}(uint16_t* out, uint16_t* in) {{
    reorder_{dims}_{pfmt}_{bfmt}(out, in);
}}

inline void unpack_{name}(uint16_t* out, uint16_t* in) {{
    reorder_{dims}_{bfmt}_{pfmt}(out, in);
}}
'''
    lines = []
    for name, details in sort_dict(arg_name_to_format):
        dims = details.logical_dims
        if details.format.is_plain:
            size = 1
            for v in dims:
                size *= v
            line = plain_templ.format(name=name, num_elem=size)
        else:
            line = block_templ.format(name=name, dims=dims_to_str(
                dims), pfmt=pysc.data_format.get_plain_by_dims(len(dims)), bfmt=details.format)
        lines.append(line)

    return "\n".join(lines)


def compile_graph(g: pysc.graph.graph, ctx):
    reorder_to_add = dict()
    ops_args = []
    for op in g:
        if op.op_name == "output":
            ops_args.append(op)
            detail = op.inputs[0].details
            plain_fmt = pysc.data_format.get_plain_by_dims(
                len(detail.logical_dims))
            name = str(detail) + str(plain_fmt)
            if not detail.format.is_plain:
                if name not in reorder_to_add:
                    reorder_to_add[name] = (detail, plain_fmt)
    for op in g:
        if op.op_name == "input":
            ops_args.append(op)
            detail = op.outputs[0].details
            plain_fmt = pysc.data_format.get_plain_by_dims(
                len(detail.logical_dims))
            target_detail = pysc.graph.tensor_detail(
                plain_fmt, detail.logical_dims, detail.dtype)
            name = str(target_detail) + str(detail.format)
            if not detail.format.is_plain:
                if name not in reorder_to_add:
                    reorder_to_add[name] = (target_detail, detail.format)
    #ops_args.reverse()
    gl = g.lower(ctx, ops_args)

    gname = g.attrs["temp.name"].extract()
    print("Graph name", gname)
    doc_src_list = []
    for op in ops_args:
        if op.op_name == "input":
            detail = op.outputs[0].details
        elif op.op_name == "output":
            detail = op.inputs[0].details
        detail_str = str(detail)
        op_name = get_op_name(op)
        line = " * @param {op_name} {inout} tensor, {detail}".format(
            op_name=op_name, inout=op.op_name, detail=detail_str)
        if not detail.format.is_plain:
            line += ", real dims is " + str(detail.real_dims)
        doc_src_list.append(line)
    doc_src = '''
/**
 * {gname}
{lines}
**/
'''.format(gname=gname, lines="\n".join(doc_src_list))

    jit = pysc.jit.cfake_jit(ctx)
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    src = jit.codegen_to_cpp(gl, False, "tmp/mod_raw.bin")
    # print(gl)

    # reorder_g.attrs["is_input_plain"] = False
    # reorder_g.attrs["is_output_plain"] = False
    # reorder_g.run_passes_and_tune(ctx, timeout=0)
    # print(reorder_g)
    return src, doc_src, gname, reorder_to_add


def make_reorder_graph(reorder_to_add, ctx):
    reorder_g = pysc.graph.graph()
    sorted_map = sort_dict(reorder_to_add)
    reorder_ins = [kv[1][0] for kv in sorted_map]
    reorder_fmts = [kv[1][1] for kv in sorted_map]
    in_tsr = reorder_g.make_input(reorder_ins).outputs
    outs = []
    reorder_info = []
    for idx, t in enumerate(in_tsr):
        otsr = reorder_g.make("reorder", [t], [], pysc.any_map(
            {"out_format": reorder_fmts[idx], "internal": True})).outputs[0]
        outs.append(otsr)
        func_name = "reorder_{dims}_{ifmt}_{ofmt}".format(
            ifmt=reorder_ins[idx].format, ofmt=reorder_fmts[idx], dims=dims_to_str(reorder_ins[idx].logical_dims))
        reorder_info.append(
            (reorder_ins[idx], reorder_fmts[idx], func_name))
    reorder_g.make_output(outs)
    print(reorder_g)
    gl = reorder_g.lower(ctx)
    jit = pysc.jit.cfake_jit(ctx)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    src = jit.codegen_to_cpp(gl, False, "tmp/mod_raw.bin")
    doc_template = '''
/**
 * Do reorder from {ifmt} format to {ofmt} format. Plain dims is {dims}
**/
extern "C" bool {func_name}(uint16_t* out, uint16_t* in);
'''
    doc_src = '\n'.join([doc_template.format(
        ifmt=info[0].format, ofmt=info[1], func_name=info[2], dims=dims_to_str(info[0].logical_dims)) for info in reorder_info])
    return src, doc_src, reorder_info


def process_global_data(graph_name: str, out_file_name: str, no_data_source: bool):
    with open("tmp/mod_raw.bin", 'rb') as binf:
        magic = struct.unpack('Q', binf.read(8))[0]
        assert(magic == 0xC0FFEEC011001010)
        total_size = struct.unpack('Q', binf.read(8))[0]
        print("Module data total size =", total_size)
        inited_size = struct.unpack('Q', binf.read(8))[0]
        print("Module data initialized size =", inited_size)
        data = binf.read(inited_size)
        inited_size = int(math.ceil(inited_size/64.0))*64
        if(total_size < inited_size):
            uninited_size = 0
        else:
            uninited_size = total_size-inited_size
        with open("tmp/mod.bin", 'wb') as boutf:
            boutf.write(data)
            # boutf.write(bytearray(total_size-inited_size))
        if not no_data_source:
            len_data = len(data)
            if len_data % 8 != 0:
                padding = (len_data//8 + 1) * 8 - len_data
                data += bytearray(padding)
                len_data = len(data)
            with open(out_file_name, 'w') as outf:
                outf.write(
                    '#include <stdint.h>\nalignas(64) uint64_t {name}_data[]={{\n'.format(name=graph_name))
                for i in range(len_data//8):
                    val = struct.unpack('Q', data[i*8:(i+1)*8])[0]
                    outf.write(hex(val)+",")
                outf.write("};")
        return (inited_size, uninited_size)


def is_closure_wrapper(line):
    if "_closure_" in line and "_0wrapper" in line:
        return True
    return False

def process_source(src: str,
                   graph_name: str,
                   inited_size: int,
                   uninited_size: int,
                   out_file_name: str,
                   replace_map: list = []):
    src_header = '''
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;
'''
    new_header = src_header + \
        '''
extern int8_t {name}_data[{init_size}];
static constexpr int8_t* __module_data = {name}_data;
alignas(64) static int8_t __uninitialized_data[{size}UL];'''.format(
            name=graph_name, size=uninited_size, init_size=inited_size)
    new_lines = []
    module_data_use = ")&__module_data["
    module_data_use_end = "UL];"
    need_remove = False
    decl = ""
    wrapper_func_suffix = "_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{"
    main_entry_prefix = 'extern "C" void main_entry('
    for index, line in enumerate(src.split("\n")):
        if index == 0:
            new_lines.append(new_header)
            continue
        if "_fptr)" in line:
            continue
        arg_remove = "(void* __stream, int8_t* __restrict__ __module_data, "

        if (line.endswith(wrapper_func_suffix) or line.startswith(main_entry_prefix)) and "_closure_" not in line:
            # remove wrappers
            need_remove = True
            continue
        elif need_remove:
            if line == "}":
                # end of wrappers
                need_remove = False
            continue
        elif arg_remove in line and not is_closure_wrapper(line):
            line = line.replace(arg_remove, "(")

            # update nonnull idx.
            pattern = re.compile(r'noexcept __attribute__\(\(nonnull \((.+)\)\)\)')
            attr_pattern = re.search(pattern, line)
            if attr_pattern:
                original_idx = attr_pattern.group(1)
                new_idx = [int(x) - 2 for x in original_idx.split(',') if (int(x) - 2) > 0]
                new_idx_str = ','.join(str(x) for x in new_idx)
                line = line.replace(original_idx, new_idx_str)
        elif line.startswith(
                'extern "C" void __sc_init__(void* __stream, int8_t* __restrict__ __module_data)'
        ):
            if line[-1] == ';':
                continue
            else:
                line = 'extern "C" void sc_init_{}() {{'.format(graph_name)
        else:
            idx = line.find(module_data_use)
            if idx != -1:
                number_idx_start = idx + len(module_data_use)
                end_idx = line.find(module_data_use_end, number_idx_start)
                assert (end_idx != -1)
                offset_str = line[number_idx_start:end_idx]
                offset = int(offset_str)
                if offset >= inited_size:
                    line = line[:idx] + ")&__uninitialized_data[" + \
                        str(offset-inited_size)+module_data_use_end

        to_replace = [
            ("(__stream, __module_data, ", "("),
            ("sc_aligned_malloc_fptr", "sc_aligned_malloc"),
            ("sc_aligned_free_fptr", "sc_aligned_free"),
            ("sc_parallel_call_cpu_with_env_fptr",
             "sc_parallel_call_cpu_with_env"),
            ("_fptr(", "("),
            ("sc_get_thread_id", "omp_get_thread_num"),
            ("sc_init_barrier", "sc_init_barrier_call"),
            ("sc_arrive_at_barrier", "sc_arrive_at_barrier_call"),
        ]
        for arg_remove, arg_new in to_replace:
            if arg_remove in line:
                line = line.replace(arg_remove, arg_new)
        for arg_remove, arg_new in replace_map:
            if arg_remove in line:
                line = line.replace(arg_remove, arg_new)
        if line.startswith('extern "C" void ' + graph_name +
                           '(') and line[-1] == '{':
            decl = line[:-1]+';' + '\n\n' + \
                'extern "C" void sc_init_{}();'.format(graph_name) + "\n"
        if line.startswith('extern "C" bool '):
            line = line.replace('extern "C" bool ', "static bool ")
        new_lines.append(line)

    with open(out_file_name, 'w') as f:
        f.write("\n".join(new_lines))
    return decl
