  export WORKDIR=$PWD
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

  if [[ -n $code && -d ${code} ]];then
     REPODIR=$code
  fi

  echo "Install loadgen"
  git clone https://github.com/mlcommons/inference.git
  cd inference && git checkout r2.1
  git log -1
  git submodule update --init --recursive
  cd loadgen
  CFLAGS="-std=c++14" python setup.py install
  cd ..; cp ${WORKDIR}/inference/mlperf.conf ${REPODIR}/closed/Intel/code/dlrm-99.9/pytorch-cpu/.

  echo "Clone source code and Install"
  echo "Install Intel Extension for PyTorch"
  cd ${WORKDIR}
  # clone Intel Extension for PyTorch
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  cd intel-extension-for-pytorch
  git checkout 1.9.0-rc
  git submodule sync
  git submodule update --init --recursive
  git log -1
  cd third_party/mkl-dnn/
  git checkout b5e06126da38bd8bee609d2965d62d30a53fe6b9
  cd ../../
  git apply ${REPODIR}/closed/Intel/calibration/dlrm-99.9/pytorch-cpu/dlrm.diff
  python setup.py install
  cd ..
