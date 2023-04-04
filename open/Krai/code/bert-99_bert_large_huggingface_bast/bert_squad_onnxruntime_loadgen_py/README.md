# MLPerf Inference - BERT - SQUAD 1.1 - ONNX

This Python implementation runs ONNX models for BERT.

Currently it supports the following models:
- bert_large on SQUAD 1.1 dataset

## Prerequisites

This workflow is designed to showcase the `axs` workflow management system.
So the only prerequisite from the user's point of view is a sufficiently fresh version of `axs` system.

First, clone the `axs` repository.
```
git clone https://github.com/krai/axs
```

Then, add the path to `bashrc`.
```
echo "export PATH='$PATH:$HOME/axs'" >> ~/.bashrc && \
source ~/.bashrc
```

Finally, import this repository into your `work_collection`
```
axs byquery git_repo,collection,repo_name=axs2mlperf
```

The dependencies of various components (on Python code and external utilities) as well as interdependencies of the workflow's main components (original dataset, preprocessed dataset, model and its parameters) have been described in `axs`'s internal language to achieve the fullest automation we could.

Please note that due to this automation (automatic recursive installation of all dependent components) the external timing of the initial runs (when new components have to be downloaded and/or installed) may not be very useful. The internal timing as measured by the LoadGen API should be trusted instead, which is not affected by these changes in external infrastructure.


## Initial clean-up (optional)

In some cases it may be desirable to "start from a clean slate" - i.e. clean up all the cached `axs` entries,
which includes the model with weights, the original COCO dataset and its resized versions
(different models need different resizing resolutions), as well as all the necessary Python packages.

On the other hand, since all those components may take considerable time to be installed, we do not recommend cleaning up between individual runs.
The entry cache is there for a reason.

The following command effectively wipes off hours of downloading, compilation and/or installation:
```
axs work_collection , remove && \
axs byquery git_repo,collection,repo_name=axs2mlperf
```

## Performing a short Accuracy run (some parameters by default)

The following test run should trigger downloading and installation of the necessary Python packages, the default model (bert_large), the SQUAD dataset and the default dataset size(loadgen_dataset_size=20):
```
axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline , get accuracy
```


## Performing a short Accuracy run (specifying the number of samples to run on)

The following test run should trigger downloading and installation of the necessary Python packages, the default model (bert_large), the SQUAD dataset and a short partial resized subset of 20 images:
```
axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=20 , get accuracy
```
The accuracy value should be printed after a successful run.


## Performing a short Accuracy run (specifying the model)

The following test run should trigger (in addition to the above) downloading and installation of the bert_large model:
```
axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=20,model_name=bert_large , get accuracy
```
The f1 value should be printed after a successful run.


## Benchmarking bert_large model in the Accuracy mode

The following command will run on the whole dataset of 10833 images used by the bert_large model. Please note that depending on whether both the hardware and the software supports running on the GPU, the run may be performed either on the GPU or on the CPU. For running on the CPU it is necessary to add execution_device=cpu to the command.
(There are ways to constrain this to the CPU only.)
```
time axs byquery loadgen_output,bert_squad,framework=onnx,model_name=bert_large,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=10833,loadgen_buffer_size=100 , get accuracy
```
The f1 value and running time should be printed after a successful run.
<details><pre>
...
90.88232621193973
                                                                                                                                                                                            real    3m31.117s
</pre></details>


## Benchmarking bert_large model in the Performance mode

###Offline

Two important changes for performance mode should be taken into account:
1. There is no way to measure f1 (LoadGen's constraint)
2. You need to "guess" the `loadgen_target_qps` parameter, from which the testing regime will be generated in order to measure the actual QPS.

So `TargetQPS` is the input, whereas `QPS` is the output of this benchmark:
```
axs byquery loadgen_output,bert_squad,framework=onnx,model_name=bert_large,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=65,loadgen_dataset_size=10833,loadgen_buffer_size=10833 , get performance
```
Measured QPS:
```
...
65.6684
```

###SingleStream

You need to set the `loadgen_target_latency` parameter.
```
axs byquery loadgen_output,bert_squad,framework=onnx,model_name=bert_large,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_latency_qps=15,loadgen_dataset_size=10833,loadgen_buffer_size=10833 , get performance
```
```
...
15364851
```
