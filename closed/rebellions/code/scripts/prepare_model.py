from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("type", type=str, choices=["resnet50", "bert_large"])
parser.add_argument("scenario", type=str, choices=["SingleStream", "MultiStream"])
parser.add_argument("--import-path", type=Path, default=None)
parser.add_argument(
    "--model-path",
    type=Path,
    help="(Path) type, \
        Path to reference quantized graph file.",
)

args = parser.parse_args()

import_path: Path = args.import_path
model_path: Path = args.model_path

if args.scenario == "SingleStream":
    MODEL_SCRATCH_PATH = Path(f"scratch/models/{args.type}_ss")
elif args.scenario == "MultiStream":
    MODEL_SCRATCH_PATH = Path(f"scratch/models/{args.type}_ms")
else:
    raise NotImplementedError()

MODEL_SCRATCH_PATH.mkdir(parents=True, exist_ok=True)


def compile_resnet50():
    import torch
    import torch.nn as nn
    from rebel_algorithms.graph_view.torch_impl import TorchGraphView
    import rebel

    module = torch.load(model_path)
    module.eval()
    batch_size = 1 if args.scenario == "SingleStream" else 8

    @torch.no_grad()
    def replace_conv1(pm):
        original_conv = pm.conv1
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.fill_(0)
        new_conv1.weight.data[:, :3] = original_conv.weight.data
        pm.conv1 = new_conv1
        return pm

    # Set first convolution to 4-channels
    replace_conv1(module)

    graph = TorchGraphView(
        module,
        None,
        [("x", torch.randn(batch_size, 4, 224, 224, dtype=torch.float32))],
    )

    # Eliminate first quantization (int8 input should be given in runtime)
    nodes = [n for n in graph.fx_graph.nodes]
    x = nodes[0]
    conv1 = nodes[9]
    conv1.args = (x,)
    x.next.append(conv1)
    graph.fx_graph.eliminate_dead_code()
    graph.refresh()

    # Export to relay
    mod, params = graph.export_to_relay(input_types="int8")

    # Build
    return rebel.build(
        mod,
        batch_size=batch_size,
        target="llvm",
        params=params,
        allow_op_fusion=True,
        compile_type="rebel_descriptor",
    )


def compile_bert():
    import tensorflow as tf
    from rebel_algorithms.graph_view.tf_impl import TFGraphView
    import rebel

    # Load model
    with open(model_path, "rb") as f:
        tf_graph_def = tf.GraphDef()
        tf_graph_def.ParseFromString(f.read())

    graph = TFGraphView(tf_graph_def, "logits")
    mod, params = graph.export_to_relay()
    return rebel.build(
        mod,
        target="llvm",
        params=params,
        allow_op_fusion=True,
        compile_type="rebel_descriptor",
    )


if import_path is not None:
    graph_filename = import_path.name + ".graph"
    graph_src_path, graph_dst_path = (
        import_path.parent / graph_filename,
        MODEL_SCRATCH_PATH / graph_filename,
    )
    graph_dst_path.unlink(missing_ok=True)
    graph_dst_path.symlink_to(graph_src_path)

    lib_filename = import_path.name + ".so"
    lib_src_path, lib_dst_path = (
        import_path.parent / lib_filename,
        MODEL_SCRATCH_PATH / lib_filename,
    )
    lib_dst_path.unlink(missing_ok=True)
    lib_dst_path.symlink_to(lib_src_path)

    params_filename = import_path.name + ".params"
    params_src_path, params_dst_path = (
        import_path.parent / params_filename,
        MODEL_SCRATCH_PATH / params_filename,
    )
    params_dst_path.unlink(missing_ok=True)
    params_dst_path.symlink_to(params_src_path)
else:
    if args.type == "resnet50":
        graph, compiled_lib, params = compile_resnet50()
    elif args.type == "bert":
        graph, compiled_lib, params = compile_bert()
    else:
        raise NotImplementedError()

    import rebel

    rebel.save(str(MODEL_SCRATCH_PATH / "RebelRuntime"), graph, compiled_lib, params)
