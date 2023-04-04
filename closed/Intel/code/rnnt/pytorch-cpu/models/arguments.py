import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perf_count", type=int, default=None, help="number of samples"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--split_len",
        type=int,
        default=-1,
        help="max splitted sequence length(value=even)",
    )
    parser.add_argument(
        "--scenario",
        default="Offline",
        choices=["SingleStream", "Offline", "Server"],
        help="Scenario",
    )
    parser.add_argument(
        "--mlperf_conf", default="../configs/mlperf.conf", help="mlperf rules config"
    )
    parser.add_argument(
        "--user_conf",
        default="../configs/user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--toml_path",
        type=str,
        default="../configs/rnnt.toml",
        help="rnn-t model parameters config",
    )
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--processor_path", type=str, default="work_dir/processpr.pt")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--infer_dataset_dir", type=str)
    parser.add_argument("--calib_dataset_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument(
        "--run_mode",
        default=None,
        choices=[None, "f32", "calib", "quant", "fake_quant"],
        help="run_mode, default None(fp32)",
    )
    parser.add_argument(
        "--enable_bf16", action="store_true", help="enable bf16 for prediction & joint"
    )
    parser.add_argument("--load_jit", action="store_true", help="load jit model")
    parser.add_argument("--save_jit", action="store_true", help="save jit model")
    parser.add_argument(
        "--enable_process", action="store_true", help="enable audio process"
    )
    parser.add_argument(
        "--accuracy", action="store_true", help="enable accuracy evaluation"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="enable benchmark evaluation"
    )
    parser.add_argument(
        "--calibration", action="store_true", help="enable model calibration"
    )
    args = parser.parse_args()
    print(args)
    return args
