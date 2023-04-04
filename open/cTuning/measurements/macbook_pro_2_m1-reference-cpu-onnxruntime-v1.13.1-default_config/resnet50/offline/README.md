This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission,_full \
	--implementation=reference \
	--model=resnet50 \
	--backend=onnxruntime \
	--device=cpu \
	--execution_mode=valid \
	--push-to-github=on \
	--results_dir=/Users/arjun/inference_3.0_results \
	--results_git_url=https://github.com/arjunsuresh/mlperf_inference_submissions_v3.0 \
	--clean \
	--readme=yes
```
## Dependent CM scripts 

1. `cm run script --tags=detect,os`

2. `cm run script --tags=detect,cpu`

`cm run script --tags=get,sys-utils-cm`

`cm run script --tags=get,python`

`cm run script --tags=get,mlcommons,inference,src`

`cm run script --tags=get,dataset-aux,imagenet-aux`

## Dependent CM scripts for the MLPerf Inference Implementation

`cm run script --tags=detect,os`

`cm run script --tags=detect,cpu`

`cm run script --tags=get,sys-utils-cm`

`cm run script --tags=get,python`

`cm run script --tags=get,generic-python-lib,_onnxruntime`

`cm run script --tags=get,ml-model,image-classification,resnet50,_onnx,raw,_fp32`

`cm run script --tags=get,dataset,image-classification,imagenet,preprocessed,_NCHW,_full`

`cm run script --tags=get,dataset-aux,image-classification,imagenet-aux`

`cm run script --tags=generate,user-conf,mlperf,inference`

`cm run script --tags=get,loadgen`

`cm run script --tags=get,mlcommons,inference,src`

`cm run script --tags=get,generic-python-lib,_opencv-python`

`cm run script --tags=get,generic-python-lib,_numpy`

`cm run script --tags=get,generic-python-lib,_pycocotools`
