# Install GCC11.2
conda install -c conda-forge gxx=11.2 libunwind --yes

#setup env for GCC11
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++
export LIBRARY_PATH=${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib:$WORKDIR/anaconda3/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
export LD_RUN_PATH=$LD_LIBRARY_PATH:$LD_RUN_PATH
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH}
