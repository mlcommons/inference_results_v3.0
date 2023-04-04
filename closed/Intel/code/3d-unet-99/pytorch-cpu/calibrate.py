import argparse
import sys
import os
import json
sys.path.insert(0, os.path.join(os.getcwd(),"common"))
from configParser import parseWorkloadConfig
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("3dunet-calibrate")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_to_calibrate",
                        type=int,
                        default=4,
                        help="number of samples used for calibration")
    parser.add_argument("--workload-config", 
                        help="A json file that contains Workload related arguments for creating sut and dataset instances",
                        default="config_calibrate.json")
    args = parser.parse_args()
    return args

def post_process(file_path):
    with open(file_path) as f:
        values = json.load(f)

        for i in range(1, len(values)):
            if values[i]["name"] == "Conv3d" and values[i - 1]["name"] == "Deconv3d":
                values[i - 1]["outputs_scale"] = values[i]["inputs_scale"]

    with open(file_path, 'w') as f:
        json.dump(values, f, indent=4)

def main():
    args = get_args()

    # Imports
    from Dataset import Dataset
    from Backend import Backend

    # Get workload config parameters
    backend_params, dataset_params, enqueue_params, buckets, num_resp_qs, import_path = parseWorkloadConfig(args.workload_config)

    # Create dataset and backend objects
    log.info("Creating backend object")
    backendObj = Backend(**backend_params)

    log.info("Creating dataset object")
    datasetObj = Dataset(**dataset_params)

    backendObj.load_model()
    samples=[i for i in range(args.samples_to_calibrate)]
    backendObj.calibrate(samples, datasetObj.get_qsl(), 'calibration_result.json')

    post_process('calibration_result.json')


if __name__ == "__main__":
    main()
