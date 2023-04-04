#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

set_one_ccd_flag () {
  local ccdflagfile=$1
  if [ -f $ccdflagfile ]; then
    sudo /home/scratch.computelab/sudo/scelnx /i /s $ccdflagfile
    sudo /home/scratch.computelab/sudo/scelnx /o /s /dev/stdout | grep -A9 'CCD Control' | head
  else
    echo "Cannot find $ccdflagfile"
  fi
}

set_4_ccd () {
  set_one_ccd_flag "power/bios_setting/4_ccds.txt"
}

set_all_ccd () {
  set_one_ccd_flag "power/bios_setting/all_ccds.txt"
}
