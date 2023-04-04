# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

SUBMITTER=NVIDIA
TARBALL_NAME=mlperf_submission_${SUBMITTER}.tar.gz
SHA1_FILE_NAME=mlperf_submission_${SUBMITTER}.sha1
SUBMISSION_LOG_SUMMARY=mlperf_submission_${SUBMITTER}_checker_summary.txt

if [ "$1" = "--pack" ]; then
    echo "Packing tarball and encrypting"
    tar -cvzf - closed/ | openssl enc -e -aes256 -out ${TARBALL_NAME}
    echo "Generating sha1sum of tarball"
    sha1sum ${TARBALL_NAME} | tee ${SHA1_FILE_NAME}
    # Grab the last 2 lines of the submission checker
    if [[ -f closed/NVIDIA/results/submission_checker_log.txt ]]; then
        tail -n 2 closed/NVIDIA/results/submission_checker_log.txt > ${SUBMISSION_LOG_SUMMARY}
    else
        echo "Could not find submission_checker_log.txt"
    fi
elif [ "$1" = "--unpack" ]; then
    echo "Checking sha1sum of tarball"
    if [ "`sha1sum ${TARBALL_NAME}`" = "`cat ${SHA1_FILE_NAME}`" ]; then
        echo "sha1sum matches."
        openssl enc -d -aes256 -in ${TARBALL_NAME} | tar -xvz
    else
        echo "ERROR: sha1sum of ${TARBALL_NAME} does not match contents of ${SHA1_FILE_NAME}"
    fi
else
    echo "Unrecognized flag. Must specify --pack or --unpack"
fi

