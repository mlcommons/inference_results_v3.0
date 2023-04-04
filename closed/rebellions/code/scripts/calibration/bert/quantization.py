import argparse
from pathlib import Path
import os

import numpy as np
from rebel_algorithms.graph_view.tf_impl import TFGraphView
from rebel_algorithms.algorithms.calibration import calibration, CALIBRATION_METHODS
import tensorflow as tf

from model import BertConfig, BertModel
from dataset import TFSQuAD

seed_ = 0
tf.random.set_random_seed(seed_)
os.environ["PYTHONHASHSEED"] = str(seed_)
np.random.seed(seed_)


def PTQ(
    graph: TFGraphView, quant_bit=8, calibration_method=None, dataset=None, model="bert"
) -> TFGraphView:

    assert quant_bit == 8
    assert calibration_method is not None
    assert dataset is not None
    assert model == "bert"

    # Select edges to Q or dQ

    matmul_nodes = graph[lambda x: x.type in ["MatMul", "BatchMatMulV2"]]
    matmul_inputs = list()
    for matmul_node in matmul_nodes:
        matmul_input = matmul_node.inbounds[0]
        if matmul_input not in matmul_inputs:
            matmul_inputs.append(matmul_input)

    features = [f for f in dataset]

    for matmul_input in matmul_inputs:

        session = graph.generate()
        outbound_nodes = [v for v in matmul_input.outbounds]  # Cache it before refresh
        print("Calibrating", matmul_input.name)

        if matmul_input.name.find("Softmax") != -1:
            print(f"Found Softmax {matmul_input}, forced scale factor 1/127...")
            activation_scale_factor = 1 / 127.0
        else:
            activations = list()
            for feature in features:
                activaion = session.run(
                    matmul_input.name,
                    {
                        "input_ids:0": np.reshape(feature.input_ids, [1, 384]),
                        "input_mask:0": np.reshape(feature.input_mask, [1, 384]),
                        "segment_ids:0": np.reshape(feature.segment_ids, [1, 384]),
                    },
                )
                activations.append(activaion)

            activation_scale_factor = calibration(
                activations, 8, method="percentile", channelwise=False, percentile=99.999
            )
            print(f"Actvtn : {activation_scale_factor}")

        graph.quantize(matmul_input, 8, activation_scale_factor)

        for outbound_node in outbound_nodes:
            # print("# of nodes : ", len(graph.tf_graph_def.node))
            if outbound_node.type in ["MatMul"]:

                node_def = [v for v in graph.tf_graph_def.node if v.name == outbound_node.name][0]
                transpose_b = node_def.attr["transpose_b"].b
                if transpose_b:
                    channel_index = 0
                else:
                    channel_index = 1

                weights = session.run(outbound_node.inbounds[1].name)
                weights_scale_factor = calibration(
                    weights,
                    8,
                    method="l2",
                    channelwise=True,
                    channel_axis=channel_index,
                )

                graph.quantize(
                    outbound_node.inbounds[1],
                    8,
                    weights_scale_factor,
                )
                print(f"Weight : [{outbound_node}, {weights_scale_factor.flatten()}]")

                if transpose_b and not isinstance(weights_scale_factor, float):
                    weights_scale_factor = np.transpose(weights_scale_factor)

                graph.dequantize(
                    outbound_node.outbounds[0],
                    activation_scale_factor * weights_scale_factor,
                )
            elif outbound_node.type in ["BatchMatMulV2"]:
                if outbound_node.inbounds[1].name.find("Softmax") != -1:
                    print(
                        f"Found Softmax for BatchMM {outbound_node.inbounds[1]}, forced scale factor 1/127. \n"
                    )
                    activation2_scale_factor = 1 / 127.0
                else:
                    activations = list()
                    for feature in features:
                        activaion = session.run(
                            outbound_node.inbounds[1].name,
                            {
                                "input_ids:0": np.reshape(feature.input_ids, [1, 384]),
                                "input_mask:0": np.reshape(feature.input_mask, [1, 384]),
                                "segment_ids:0": np.reshape(feature.segment_ids, [1, 384]),
                            },
                        )
                        activations.append(activaion)

                    activation2_scale_factor = calibration(
                        activations, 8, method="percentile", channelwise=False, percentile=99.999
                    )
                    print(f"Actvtn2 : {activation2_scale_factor}")

                graph.quantize(outbound_node.inbounds[1], 8, activation2_scale_factor)
                graph.dequantize(
                    outbound_node.outbounds[0],
                    activation_scale_factor * activation2_scale_factor,
                )
            else:
                target_edge = outbound_node.inbounds[1]
                graph.dequantize(
                    target_edge,
                    activation_scale_factor,
                    [v for v in target_edge.outbounds if v is not outbound_node],
                )

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="(str) type, Path to the TF checkpoint (pretrained model)"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="(Path) type, Path to the dataset.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        choices=list(CALIBRATION_METHODS),
        default="l2",
        help="(str) type, \
            Calibration methods",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=".",
        help="(Path) type, Path to output file be stored.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If true, preprocessing will be executed without saving graph(pb2).",
    )
    parser.add_argument("--pb2", type=str, default="", help="output pb graph filename.")
    args = parser.parse_args()

    # Load model
    bert_config_path = "bert_config.json"
    bert_config_path = os.path.join("configs", bert_config_path)
    bert_config = BertConfig.from_json_file(bert_config_path)

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=(1, 384), name="input_ids")
        input_mask = tf.placeholder(tf.int32, shape=(1, 384), name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=(1, 384), name="segment_ids")
        model = BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False,
            compute_type=tf.float32,
        )
        final_hidden = model.get_sequence_output()
        final_hidden_shape = final_hidden.shape.as_list()
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        output_weights = tf.get_variable(
            "cls/squad/output_weights",
            [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer()
        )
        final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        predictions = tf.reshape(logits, [batch_size, seq_length, 2], name="logits")
        output_node_name = predictions.op.name

        # Freeze graph
        with tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1,
                allow_soft_placement=True,
            ),
            graph=tf_graph,
        ) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(reshape=True)
            saver.restore(sess, args.model_path)
            tf_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, tf_graph.as_graph_def(), [output_node_name]
            )

    # Define a graph
    graph = TFGraphView(tf_graph_def, output_node_name)

    # quantize
    dataset = TFSQuAD(args.data_path)
    graph = PTQ(
        graph=graph,
        quant_bit=8,
        calibration_method=args.calibration_method,
        dataset=dataset,
        model="bert",
    )

    if args.pb2 == "":
        pb_filename = args.output_path / f"bert_large_8bit_def.pb2"
    else:
        pb_filename = args.pb2

    if not args.dry_run:
        with open(pb_filename, "wb") as f:
            f.write(graph.tf_graph_def.SerializeToString())
