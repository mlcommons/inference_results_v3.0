This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_populate-readme,_all-scenarios \
	--model=bert-99 \
	--device=cuda \
	--implementation=reference \
	--backend=tf \
	--execution-mode=valid \
	--results_dir=/home/rsa-key-fgg-universal/inference_3.0_results \
	--quiet
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


5.  `cm run script --tags=get,dataset,squad,language-processing`


6.  `cm run script --tags=get,dataset-aux,squad-vocab`

## Dependent CM scripts for the MLPerf Inference Implementation


1. `cm run script --tags=detect,os`


2. `cm run script --tags=detect,cpu`


3. `cm run script --tags=get,sys-utils-cm`


4. `cm run script --tags=get,python`


5. `cm run script --tags=get,cuda,_cudnn`


6. `cm run script --tags=get,generic-python-lib,_torch`


7. `cm run script --tags=get,generic-python-lib,_transformers`


8. `cm run script --tags=get,generic-python-lib,_tensorflow`


9. `cm run script --tags=get,ml-model,language-processing,bert-large,_fp32,raw,_tf`


10. `cm run script --tags=get,dataset,squad,original`


11. `cm run script --tags=get,dataset-aux,squad-vocab`


12. `cm run script --tags=generate,user-conf,mlperf,inference`


13. `cm run script --tags=get,loadgen`


14. `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


15. `cm run script --tags=get,generic-python-lib,_tokenization`


16. `cm run script --tags=get,generic-python-lib,_six`


17. `cm run script --tags=get,generic-python-lib,_protobuf`
