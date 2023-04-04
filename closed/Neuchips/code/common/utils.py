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

import os
import re
import importlib
import importlib.util

from typing import Any, Dict, List, Optional

# Conditional import. Sometimes, we may use the constants files outside of an environment that has numpy installed, for
# example certain scripts in CI/CD. Provide an environment variable 'OUTSIDE_MLPINF_ENV' to allow using constants.py
# outside of the docker.
if importlib.util.find_spec("numpy") is not None or os.environ.get("OUTSIDE_MLPINF_ENV", "0") == "0":
    import numpy as np

    def get_dyn_ranges(cache_file: str) -> Dict[str, np.uint32]:
        """
        Get dynamic ranges from calibration file for network tensors.

        Args:
            cache_file (str):
                Path to INT8 calibration cache file.

        Returns:
            Dict[str, np.uint32]: Dictionary of tensor name -> dynamic range of tensor
        """
        dyn_ranges = {}
        if not os.path.exists(cache_file):
            raise FileNotFoundError("{} calibration file is not found.".format(cache_file))

        with open(cache_file, "rb") as f:
            lines = f.read().decode('ascii').splitlines()
        for line in lines:
            regex = r"(.+): (\w+)"
            results = re.findall(regex, line)
            # Omit unmatched lines
            if len(results) == 0 or len(results[0]) != 2:
                continue
            results = results[0]
            tensor_name = results[0]
            # Map dynamic range from [0.0 - 1.0] to [0.0 - 127.0]
            dynamic_range = np.uint32(int(results[1], base=16)).view(np.dtype('float32')).item() * 127.0
            dyn_ranges[tensor_name] = dynamic_range
        return dyn_ranges


class Tree:
    """
    Datastructure that allows storing and retrieving values in a generic tree structure, where a node can have a
    variable number of children.
    """

    def __init__(self, starting_val: Optional[Dict[str, Any]] = None):
        if starting_val is None:
            self.tree: Dict[str, Any] = dict()
        else:
            self.tree: Dict[str, Any] = starting_val

    def insert(self, keyspace: List[str], value: Any, append: bool = False):
        """
        Inserts a value into the tree. The keyspace represents the tree traversal or walk starting from the root of the
        tree to get to the leaf. `value` is the object stored at the leaf.

        If `append` is True, then the leaf is treated as a list, and `value` is appended to it instead of overwriting
        it.

        Args:
            keyspace (List[str]):
                The tree traversal to get to the node to insert at.
            value (Any):
                The value for the inserted leaf
            append (bool):
                Default: False. If True, the leaf is treated as a list, and `value` is appended to it instead of
                overwriting.
        """
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = list(keyspace[::-1])

        curr = self.tree
        while len(keyspace) > 0:
            if len(keyspace) == 1:
                if append:
                    if keyspace[-1] not in curr:
                        curr[keyspace[-1]] = [value]
                    else:
                        if type(curr[keyspace[-1]]) is list:
                            curr[keyspace[-1]].append(value)
                        else:
                            curr[keyspace[-1]] = [curr[keyspace[-1]], value]
                else:
                    curr[keyspace[-1]] = value
            else:
                if keyspace[-1] not in curr:
                    curr[keyspace[-1]] = dict()

                curr = curr[keyspace[-1]]
            keyspace.pop(-1)

    def get(self, keyspace: List[str], default=None) -> Any:
        """
        Gets the value of a node in the tree. The keyspace represents the tree traversal or walk starting from the root
        to get to that node.

        Returns object stored at the leaf, if it exists, otherwise returns `default`.

        Args:
            keyspace (List[str]):
                The tree traversal to get to the node to insert at.

            default (Any):
                The value to return if the keyspace does not exist.

        Returns:
            Any: The value at the keyspace.
        """
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = list(keyspace[::-1])

        curr = self.tree
        while len(keyspace) > 0:
            if keyspace[-1] not in curr:
                return default

            if len(keyspace) == 1:
                return curr[keyspace[-1]]
            else:
                curr = curr[keyspace[-1]]
                keyspace.pop(-1)

    def __getitem__(self, keyspace_str):
        return self.get(keyspace_str.split(","))

    def __setitem__(self, keyspace_str, value):
        self.insert(keyspace_str.split(","), value)

    def __iter__(self):
        return (k for k in self.tree)

    def __len__(self) -> int:
        """
        Number of leaves of the tree.

        Returns:
            int: The number of leaves of the tree. If a leaf is a list or tuple of length N, then that leaf is counted N
            times (representing N different leaves).
        """
        def _count_dict(d):
            _count = 0
            for k, v in d.items():
                if isinstance(v, dict):
                    _count += _count_dict(v)
                elif any([isinstance(v, t) for t in (list, tuple)]):
                    _count += len(v)
                else:
                    _count += 1
            return _count
        return _count_dict(self.tree)
