import sys
import numpy as np
import afe
from pathlib import Path
from afe.apis.defines import QuantizationParams, default_calibration, QuantizationScheme
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from afe.ir.tensor_type import ScalarType
from afe.apis.loaded_net import load_model
from sima_utils.data.data_generator import DataGenerator
#################
# NEW API
##################

model_dir = Path(__file__).parent
model_name = "resnet50_v1"
model_file = model_dir / (model_name + ".onnx")
lm_file = model_file.with_suffix(".lm")
batch_size = 1
asym = True
per_ch = True
sample_start = 10
max_calib_samples = 35

quant_config = "a" if asym else "s"
quant_config += "c" if per_ch else "t"
mlc_model_name = f"{model_name}_b{batch_size}_{quant_config}_opt"
lm_file = model_dir / "final2" / (mlc_model_name + "_MLA_0.lm")

# Read images
cal_dat = np.fromfile(model_dir / 'calibration/mlperf_resnet50_cal_NCHW.dat', dtype=np.float32).reshape(500, 3, 224, 224)
cal_labels = np.fromfile(model_dir / 'calibration/mlperf_resnet50_cal_labels_int32.dat', dtype=np.int32)
# tranpose images from NCHW to NHWC
cal_dat_NHWC = cal_dat.transpose(0, 2, 3, 1)

params = ImporterParams(
    format=ModelFormat("onnx"),
    file_paths=[str(model_file)],
    input_names=["input_tensor:0"],
    input_types=[ScalarType.float32],
    input_shapes=[(1, 3, 224, 224)],
    layout="NCHW"
)

loaded_net = load_model(params)
print("LOADED MODEL INTO Relay IR")

quant_params = QuantizationParams(
    calibration_method=default_calibration(),
    activation_quantization_scheme=QuantizationScheme(asym, False),
    weight_quantization_scheme=QuantizationScheme(False, per_ch),
    node_names={''},
    custom_quantization_configs=None
)

dg = DataGenerator({params.input_names[0]: cal_dat_NHWC[sample_start:sample_start + max_calib_samples]})
input_generator = convert_data_generator_to_iterable(dg)
model = loaded_net.quantize(calibration_data=input_generator, quantization_config=quant_params, layout=params.layout, model_name=mlc_model_name)

model.save(model_name, output_directory=str(model_dir))
dg.set_batch_size(batch_size)
model.generate_lm_and_reference_files(
    [dg[0]], str(lm_file.parent), batch_size=batch_size, compress=True,
    tessellate_parameters={"MLA_0/placeholder_0": [[], [224], [3]]})
import subprocess
cmd = ["isim", lm_file, lm_file.with_suffix(".ifm_0.mlc"), "--check", lm_file.with_suffix(".ofm_chk_0.mlc")]
subprocess.run(cmd, check=True)
