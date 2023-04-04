set -x

WORKDIR=`pwd`

echo "Install dependencies"
echo "GCC minimum version: 11.1"
source /opt/rh/gcc-toolset-11/enable
conda install -c conda-forge cmake jemalloc --yes
conda install intel-openmp mkl mkl-include mkl-service mkl_fft mkl_random --no-update-deps --yes
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

mkdir unet3d_env
cd unet3d_env

echo "Install loadgen"
git clone https://github.com/mlcommons/inference.git
cd inference
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
cd ../..
cp ./inference/mlperf.conf .


echo "Install Intel Extension for PyTorch"
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 1.9.0-rc
git submodule sync && git submodule update --init --recursive
git apply ../../unet3d.patch
cd third_party/mkl-dnn/
git checkout v2.7
cd ../..
python setup.py install
cd ../..

bash process_data_model.sh
# python calibrate.py

