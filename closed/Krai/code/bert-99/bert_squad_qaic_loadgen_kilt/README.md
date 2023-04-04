# MLPerf Inference - Language Models - KILT
This implementation runs language models with the [KILT](https://github.com/krai/kilt-mlperf) backend.

Currently it supports the following models:
- BERT 99

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

Import these repositories into your `work_collection`
```
axs byquery git_repo,collection,repo_name=axs2mlperf && \
axs byquery git_repo,collection,repo_name=axs2kilt
```

Finally, download protobuf and the calibration dataset.
```
axs byquery compiled,protobuf && \
axs byquery tokenized,squad_v1_1,calibration=yes
```

The dependencies of various components (on Python code and external utilities) as well as interdependencies of the workflow's main components (original dataset, preprocessed dataset, model and its parameters) have been described in `axs`'s internal language to achieve the fullest automation we could.

Please note that due to this automation (automatic recursive installation of all dependent components) the external timing of the initial runs (when new components have to be downloaded and/or installed) may not be very useful. The internal timing as measured by the LoadGen API should be trusted instead, which is not affected by these changes in external infrastructure.


## Initial clean-up (optional)

In some cases it may be desirable to "start from a clean slate" - i.e. clean up all the cached `axs` entries.

On the other hand, since all those components may take considerable time to be installed, we do not recommend cleaning up between individual runs.
The entry cache is there for a reason.

The following command effectively wipes off hours of downloading, compilation and/or installation:
```
axs work_collection , remove && \
axs byquery git_repo,collection,repo_name=axs2mlperf && \
axs byquery git_repo,collection,repo_name=axs2kilt
```

## Add custom System-Under-Test (SUT)

Set the sut. Custom suts could be added by updating `sut_param_for_bert_kilt_loadgen_program/data_axs.json`.
```
export sut="chai"
```

## Performing a Offline Accuracy run

```
axs byquery loadgen_output,bert_squad,qaic,framework=kilt,model_name=bert_99,sut_name=${SUT},loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_target_qps=1,loadgen_dataset_size=10833,loadgen_buffer_size=10833 , get accuracy_dict
```
<details>
<pre>
{'exact_match': 82.53547776726585, 'f1': 90.11113134712532}
</pre>
</details>

## Performing a Offline Performance run
```
axs byquery loadgen_output,bert_squad,qaic,framework=kilt,model_name=bert_99,sut_name=${SUT},loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_target_qps=1,loadgen_dataset_size=10833,loadgen_buffer_size=10833 , get performance
```
<details>
<pre>
677.44
</pre>
</details>
