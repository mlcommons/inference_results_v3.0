import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CONFIG-PARSER")

def parseWorkloadConfig(config_file=None):
    
    if not config_file:
        log.error("Config file required")
        sys.exit(1)

    if not os.path.isfile(config_file):
        log.error("Could not find {}".format(config_file))
        sys.exit(1)

    with open(config_file,'rb') as fid:
        data = json.load(fid)
    
    backend_params = {}
    dataset_params = {}
    enqueue_params = {}
    
    backend_params = data.get('backend-params', {})
    dataset_params = data.get('dataset-params', {})
    enqueue_params = data.get('enqueue-params', {})
    bucket_params = data.get('buckets', {})
    num_out_queues = data.get('num_out_queues', 1)
    import_path = data.get('import_path', None)
    if import_path is None:
        log.error("'import_path' containing Dataset, Backend and InQueue classes need to be added to config")
        sys.exit(1)
    return backend_params, dataset_params, enqueue_params, bucket_params, num_out_queues, import_path
