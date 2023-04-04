echo "Install dependency packages"
pip install -e git+https://github.com/mlperf/logging@1.1.0-rc3#egg=mlperf-logging
pip install absl-py tqdm numpy
conda install -c conda-forge cmake gperftools  --yes
conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes
