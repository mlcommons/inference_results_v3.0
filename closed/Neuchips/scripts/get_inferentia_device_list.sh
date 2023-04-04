#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

instance_type=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
if [ $instance_type = "inf1.2xlarge" ] || [ $instance_type = "inf1.xlarge" ]
then
    echo "--device=/dev/neuron0"
fi
if [ $instance_type = "inf1.6xlarge" ]
then
    echo "--device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3"
fi
if [ $instance_type = "inf1.24xlarge" ]
then
    echo "--device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3 --device=/dev/neuron4 --device=/dev/neuron5 --device=/dev/neuron6 --device=/dev/neuron7 --device=/dev/neuron8 --device=/dev/neuron9 --device=/dev/neuron10 --device=/dev/neuron11 --device=/dev/neuron12 --device=/dev/neuron13 --device=/dev/neuron14 --device=/dev/neuron15"
fi


