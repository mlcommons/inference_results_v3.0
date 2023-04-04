#!/usr/bin/env python3

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

"""
Generates pyproject.toml and poetry.lock files from requirements.txt

Black Duck requires poetry lock files to perform security scans. TensorRT uses requirements.txt to define
Python dependencies. 
This script parses through the requirements.txt files in the project and generates the poetry lock files 
required to perform the Nspect security scans.

Pip install requirements:
pip3 install poetry

To generate pyproject.toml and poetry.lock recursively for all requirements.txt in the project:
python3 scripts/generate_lock_files.py

To generate pyproject.toml and poetry.lock for a single requirements.txt:
python3 scripts/generate_lock_files.py --path <path>/requirements.txt
"""

import argparse
import os
import subprocess
from pathlib import Path
import pkg_resources

url_mapping = {
    'nvidia-dali-cuda110==0.31.0': 'https://developer.download.nvidia.com/compute/redist/nvidia-dali-cuda110/nvidia_dali_cuda110-0.31.0-2054952-py3-none-manylinux2014_x86_64.whl',
    'torch==1.10.0+cu111': 'https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl',
    'torchvision==0.11.0+cu111': 'https://download.pytorch.org/whl/cu111/torchvision-0.11.0%2Bcu111-cp38-cp38-linux_x86_64.whl',
    'torchaudio==0.10.0': 'https://download.pytorch.org/whl/cpu/torchaudio-0.10.0%2Bcpu-cp38-cp38-linux_x86_64.whl',
    'polygraphy==0.35.2': 'https://developer.download.nvidia.com/compute/redist/polygraphy/polygraphy-0.35.2-py2.py3-none-any.whl',
    'onnx-graphsurgeon': 'https://developer.download.nvidia.com/compute/redist/onnx-graphsurgeon/onnx_graphsurgeon-0.3.10-py2.py3-none-any.whl',
    'uff': 'https://developer.download.nvidia.com/compute/redist/uff/uff-0.6.9-py2.py3-none-any.whl',

}

if __name__ == "__main__":
    """
    Sample command: python3 scripts/create_poetry_lock.py -p docker/requirements.x86_64.1.txt docker/requirements.x86_64.2.txt -o docker/

    HACK for poetry:
    If the command reports error, please change the file:
    sudo vim /home/<username>/.local/lib/python3.8/site-packages/poetry/console/application.py

    Add main() function and move Application().run() from the last line to main() function.
    """
    parser = argparse.ArgumentParser(
        description="Lock files generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--path", nargs="+",
                        help="Path, or a list of path to requirements.txt",
                        required=True)
    parser.add_argument("-o", "--output_dir",
                        help="Output directory of the toml and lock file",
                        default="docker/")
    args, _ = parser.parse_known_args()

    paths = args.path

    # Force install the poetry (because the docker doesn't install as a requirement)
    subprocess.run("sudo python3 -m pip install poetry", shell=True)

    # generate pyproject.toml and poetry.lock files in the same location
    # Combine requirements because MLPerf split them in 2 files.
    poetry_cmds = []
    for path in paths:
        output = subprocess.run(f'cat {path}', shell=True, capture_output=True, text=True).stdout
        packages = output.split('\n')

        if packages[-1] == '':  # last entry is newline
            packages = packages[:-1]

        for package in packages:
            # Hacky: ignore lines with "-f" "aarch64" or "Windows": No tool exists to parse complex requirements.txt
            if 'python_version<"3.' in package or \
                '-f' in package or \
                'aarch' in package or \
                'Windows' in package or \
                '@' in package or \
                package.startswith("#") or \
                    package.startswith('--'):
                continue

            data = package.split(';')
            package_name = data[0].replace(" ", "")
            poetry_cmd = f"poetry add '{package_name}'"
            if package_name in url_mapping:
                poetry_cmd = f"poetry add '{url_mapping[package_name]}'"
            poetry_cmds.append(poetry_cmd)

    # initiazlie poetry file in the output dir.
    toml_file = os.path.join(args.output_dir, 'pyproject.toml')
    if os.path.isfile(toml_file):
        try:
            os.remove(toml_file)
        except OSError:
            pass

    lock_file = os.path.join(args.output_dir, 'poetry.lock')
    if os.path.isfile(lock_file):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    print(f"Initializing PyProject.toml in {args.output_dir}")
    name = '"MLPerf-Inference"'
    author = '"MLPerf-Inference"'
    py_version = '"^3.8"'
    poetry_init_cmd = f'poetry init --no-interaction --name {name} --author {author} --python {py_version}'
    print(f"Running init command: {poetry_init_cmd}")
    subprocess.run(poetry_init_cmd, shell=True, cwd=args.output_dir)

    for poetry_cmd in poetry_cmds:
        print(f"{poetry_cmd}")
        subprocess.run(poetry_cmd, shell=True, cwd=args.output_dir)
