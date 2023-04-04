#! /bin/bash
set -x

WORKDIR=`pwd`
REPODIR=<path/to/this/repo>

PATTERN='[-a-zA-Z0-9_]*='
if [ $# -lt "0" ] ; then
    echo 'ERROR:'
    printf 'Please use following parameters:
    --code=<mlperf workload repo directory>
    '
    exit 1
fi
for i in "$@"
do
    case $i in
        --code=*)
	    code=`echo $i | sed "s/${PATTERN}//"`;;
        *)
	    echo "Parameter $i not recognized."; exit 1;;
    esac
done

if [ -d $code ];then
	   REPODIR=$code
fi

#conda install -c conda-forge llvm-openmp

home=${REPODIR}/closed/Intel/code/bert-99/pytorch-cpu

# use gcc-9 compile clang-15
# install clang-15
git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
git checkout llvmorg-15.0.7 
mkdir build
pushd build
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX"
ninja
ninja install
popd
popd

#install pytorch and transformers
git clone https://github.com/pytorch/pytorch pytorch
pushd pytorch
git checkout v1.12.0
git submodule sync
git submodule update --init --recursive
pushd third_party/gloo
git apply $home/patches/gloo.patch
popd
pushd third_party/ideep/mkl-dnn
git apply $home/patches/clang_mkl_dnn.patch
popd
git apply $home/patches/pytorch_official_1_12.patch
pip install -r requirements.txt
CC=clang CXX=clang++ USE_CUDA=OFF python -m pip install -e .
popd

git clone https://github.com/huggingface/transformers.git
pushd transformers
git checkout 9f4e0c23d68366985f9f584388874477ad6472d8
git apply $home/patches/transformers.patch
python -m pip install -e .
popd


# install loadgen and onednn
cd $home
git clone --recursive https://github.com/mlcommons/inference.git

pushd mlperf_plugins
git clone https://github.com/oneapi-src/oneDNN.git onednn
pushd onednn
git checkout v2.6
git apply $home/patches/onednnv2_6.patch
popd
popd

# delete MKL ERROR
rm -rf ${CONDA_PREFIX}/lib/cmake/mkl/*

mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DBUILD_TPPS_INTREE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON ..
ninja
popd

#for accuracy mode
pip install boto3 tokenization

#convert dataset and model
bash convert.sh
