import arguments
import mlperf_loadgen as lg
import os
import subprocess

from eval_accuracy import eval_acc
from pytorch_sut import PytorchSUT
from tqdm import tqdm
from utils import *


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = arguments.parse_args()
    # 1. calibration
    if args.calibration:
        sut = PytorchSUT(
            args.model_path, args.calib_dataset_dir, args.batch_size, "calib", args
        )
        print("==> Running calibration...")
        for i in tqdm(range(0, sut.qsl.count, args.batch_size)):
            results, results_idx = sut.inference(
                range(i, min(i + args.batch_size, sut.qsl.count))
            )
            for j in range(len(results)):
                logger.debug(f"{i+j}::{seq_to_sen(results[j], results_idx[j])}")
        args.model_path = os.path.join(
            os.path.dirname(args.model_path), "rnnt_calib.pt"
        )
        torch.save(sut.model.rnnt.state_dict(), args.model_path)
    # 2. jit
    if args.save_jit:
        sut = PytorchSUT(
            args.model_path,
            args.calib_dataset_dir,
            args.batch_size,
            args.run_mode,
            args,
        )
        suffix = f"_{args.run_mode}_jit" if args.save_jit else f"_{args.run_mode}"
        if args.enable_process:
            print("==> JIT audio processor...")
            args.processor_path = os.path.join(
                os.path.dirname(args.model_path), "processor_jit.pt"
            )
            torch.jit.save(sut.processor, args.processor_path)

        print("==> JIT model...")
        args.model_path = os.path.join(
            os.path.dirname(args.model_path), f"rnnt{suffix}.pt"
        )
        torch.jit.save(sut.model.rnnt, args.model_path)
    # 3. benchmark
    if args.benchmark or args.accuracy:
        sut = PytorchSUT(
            args.model_path,
            args.infer_dataset_dir,
            args.batch_size,
            args.run_mode,
            args,
        )
        # set cfg
        settings = lg.TestSettings()
        settings.scenario = scenario_map[args.scenario]
        settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
        settings.FromConfig(args.user_conf, "rnnt", args.scenario)
        settings.mode = (
            lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
        )
        # set log
        os.makedirs(args.log_dir, exist_ok=True)
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.log_dir
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        print("==> Running loadgen test")
        lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
        print("Done!")


if __name__ == "__main__":
    main()
