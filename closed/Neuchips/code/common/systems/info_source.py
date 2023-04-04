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

from __future__ import annotations
from enum import Enum, unique
from typing import Any, Callable, List


class InfoSource:
    """Utility class used to denote an information source, such as a file or command, that is consumed by other usages.
    InfoSource is served both as a cache, and is buffered, so that a command output can be stored, and consumed
    iteratively by different objects or functions."""

    def __init__(self, fn: Callable[[], List[Any]]):
        """
        Initializes an InfoSource with a function to call which retrieves and creates a buffer of data. The function is
        called once on initialization.

        Args:
            fn (Callable[[], List[Any]]):
                A function that returns a list of objects that serves as a buffer of arbitrary data units.
        """
        self.fn = fn
        self.buffer = None
        self.index = 0

        self.reset(hard=True)

    def reset(self, hard=False):
        self.index = 0
        if hard:
            self.buffer = self.fn()

    def has_next(self) -> bool:
        """
        Whether or not the buffer has more elements.

        Returns:
            bool: True if there are more items in the buffer, False otherwise
        """
        return self.index < len(self.buffer)

    def __next__(self) -> Any:
        """
        Retrieves the next item in the buffer. If the buffer is empty, raises StopIteration.

        Returns:
            Any: The next item in the buffer

        Raises:
            StopIteration: If there are no items left in the buffer
        """
        if self.has_next():
            if isinstance(self.buffer, list):
                r = self.buffer[self.index]
            elif isinstance(self.buffer, dict):
                # python 3.7 and above, keys() are deterministic (insertion order) by spec
                k = list(self.buffer.keys())[self.index]
                r = (k, self.buffer[k])
            elif isinstance(self.buffer, set):
                r = list(self.buffer)[self.index]
            else:
                raise NotImplemented
            self.index += 1
            return r
        else:
            raise StopIteration

    def __iter__(self) -> InfoSource:
        """Calls reset (soft) and returns self as an iterator.

        Returns:
            InfoSource: self as an iterator object
        """
        self.reset()
        return self
