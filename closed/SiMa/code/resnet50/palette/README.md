# SiMa Resnet50

Running SiMa's MLPerf requires purchasing our Hardware (obviously) that will copy with a copy of our SDK v0.7+.

Note that our v0.7 version of the SDK comes with MLPerf Resnet50 built into it and can run it as a "push button".

**If you proceed to try to build and run from this directory, it is possible but painful - it is highly recommended to stick to the SDK path.**


## Running

Remember that you are warned this is a long and painful path - use the SDK's builtin MLPerf instead. If you want to use this directory to build and run:
1. Download the MLPerf pytorch FP32 model and Imagenet
2. Run imgtool.py to pre-process the dataset
3. See compiling/ for code to quantize and calbirate the model - this will produce an ".lm" file, or compiled model.
4. You will need two separate model ".lm" files, one for BS=1 and one for BS=8.
5. Build the gstreamer pipeline in src/, see the readme there
6. Copy the gstreamer pipeline libraries, dataset, ".lm", config files, and scripts/ to your SiMa Dual M.2
7. Use scripts/run2 (see run_ scripts) to execute the models, choosing your scenario, mode, and batch size.
8. Collect your mlperf log files.

Note: when compiling the gstreamer plugins, you'll need to cross compile for aarch64. Our SDK provides the correct cross compiling tools and headers for you.

If you additionally want to measure power;
1. See scripts/client_ files which invoke the power client.
2. The server config is in configs






