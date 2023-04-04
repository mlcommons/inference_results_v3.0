set -e
set -x

pip install pyyaml pytest pybind11 cython pythran tornado cloudpickle decorator attrs psutil scons wheel
pip install torch==1.10.0 torchvision==0.10.0 transformers==4.24.0
pip install synr==0.6.0 tflite==2.4.0 onnx==1.10.2 tflite-runtime==2.8.0 onnxruntime==1.11.0 xgboost==1.5.0 opencv-python==4.7.0.68

apt-get install -y openssh-server
apt-get update && apt-get install -y python3-opencv

#downgrade protobuf
pip uninstall protobuf
pip install protobuf==3.20

#install rknn-toolkit-lite2
pip install /host/open/Alibaba/docker/rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl
mv /host/open/Alibaba/docker/librknnrt.so /usr/lib






