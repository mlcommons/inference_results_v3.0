echo "Install dependency packages"
pip install absl-py tqdm numpy
conda install -c conda-forge cmake gperftools --yes
conda install glog --yes
conda install -c intel intel-openmp mkl mkl-include numpy --no-update-deps --yes
conda install pytorch=1.13.1 cpuonly -c pytorch --yes
