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
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum
from numbers import Number
from typing import Callable, Final

import dataclasses
import math

from code.common.constants import *


class Matchable(ABC):
    @abstractmethod
    def matches(self, o) -> bool:
        pass

    def __eq__(self, o) -> bool:
        return self.matches(o)


class MatchAllowList(Matchable):
    """
    Utility class used to match objects against a list of various values.
    """

    def __init__(self, L):
        self.values = L

    def matches(self, o) -> bool:
        return any([o == v for v in self.values])

    def __hash__(self) -> int:
        return sum(map(hash, self.values))

    def __str__(self) -> str:
        s = "MatchAllowList("
        for v in self.values:
            s += "\n\t" + str(v)
        s += ")"
        return s

    def pretty_string(self) -> str:
        return self.codestr()

    def codestr(self) -> str:
        """Returns a string representing the line of code that constructs this instance"""
        value_str = repr(self.values)
        return f"MatchAllowList({value_str})"


class MatchAny(Matchable):
    """
    Utility class used to denote any field or object for matching that can match objects (i.e. the field is ignored
    during matching).
    """

    def matches(self, o) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("MatchAny")

    def __str__(self) -> str:
        return "MatchAny()"

    def pretty_string(self) -> str:
        return self.codestr()

    def codestr(self) -> str:
        return "MATCH_ANY"


MATCH_ANY: Final[Matchable] = MatchAny()


class MatchFloatApproximate(Matchable):
    """
    Utility class to compare 2 matchables that represent floating point numbers with an approximation.
    """

    def __init__(self, o: Matchable, to_float_fn: Callable[Matchable, float], rel_tol: float = 0.05):
        """Creates a MatchFloatApproximate given the base Matchable and a function to return a float representation of
        the Matchable.

        Args:
            o (Matchable): The object to wrap around
            to_float_fn (Callable[Matchable, float]): Function that takes a Matchable and returns a floating point
                                                      representation. The input parameter should accept the same type as
                                                      `o`.
            rel_tol (float): The relative tolerance to use for the float comparison. (Default: 0.05)
        """
        self.o = o
        self.to_float_fn = to_float_fn
        self.rel_tol = rel_tol

    def matches(self, other) -> bool:
        if self.o.__class__ == other.__class__:
            return math.isclose(self.to_float_fn(self.o), self.to_float_fn(other), rel_tol=self.rel_tol)
        elif self.__class__ == other.__class__:
            return self.o == other.o
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.o) + hash('MatchFloatApproximate')

    def __str__(self) -> str:
        return f"MatchFloatApproximate(value={self.o}, rel_tol={self.rel_tol})"

    def pretty_string(self) -> str:
        return f"approx. {self.o.quantity} {self.o.byte_suffix.name}"

    def codestr(self) -> str:
        raise ValueError("Cannot give repr representation of MatchFloatApproximate.to_float_fn")


class MatchNumericThreshold(Matchable):
    """
    Utility class to compare 2 matchables that represent numeric values, using some value as a threshold as either the
    min or max threshold.
    """

    def __init__(self, o: Matchable, to_numeric_fn: Callable[Matchable, Number], min_threshold: bool = True):
        """
        Creates a MatchNumericThreshold given the base Matchable, a function to return a Numeric representation of the
        Matchable, and whether or not the base Matchable is the minimum threshold or maximum threshold.

        Args:
            o (Matchable): The object to wrap around
            to_numeric_fn (Callable[Matchable, Number]): Function that takes a Matchable and returns a Number. The input
                                                         parameter should accept the same type as `o`.
            min_threshold (bool): If True, uses `o` as the minimum threshold when comparing, so that `self.matches`
                                  returns True if `other` is larger than `o`. Otherwise, uses `o` as the max threshold,
                                  so `self.matches` returns True if `other` is smaller than `o`. (Default: True)
        """
        self.o = o
        self.to_numeric_fn = to_numeric_fn
        self.min_threshold = min_threshold
        self.compare_symbol = ">=" if self.min_threshold else "<="

    def matches(self, other) -> bool:
        if self.o.__class__ == other.__class__:
            if self.min_threshold:
                return self.to_numeric_fn(other) >= self.to_numeric_fn(self.o)
            else:
                return self.to_numeric_fn(other) <= self.to_numeric_fn(self.o)
        elif self.__class__ == other.__class__:
            return self.o == other.o and self.min_threshold == other.min_threshold
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.o) + hash('MatchNumericThreshold')

    def __str__(self) -> str:
        return f"MatchNumericThreshold({self.compare_symbol}  {self.o})"

    def pretty_string(self) -> str:
        return f"{self.compare_symbol} {self.o.pretty_string()}"

    def codestr(self) -> str:
        raise ValueError("Cannot give repr representation of MatchNumericThreshold.to_numeric_fn")


class Hardware(Matchable):
    """
    Abstract class for representing hardware, such as a CPU or GPU, that can be matched with other hardware
    components and be detected programmatically. Subclasses of Hardware should be dataclasses by convention.
    """

    def identifiers(self):
        """Returns the identifiers used for the match behavior (__eq__) and __hash__.

        Returns:
            Tuple[Any...]: A tuple of identifiers
        """
        raise NotImplemented

    @classmethod
    def detect(cls) -> Hardware:
        raise NotImplemented

    def matches(self, other) -> bool:
        """Matches this Hardware component with 'other'.

        If other is the same class as self, compare the identifiers.
        If the other object is a MatchAny or MatchAllowList, it will use other's .matches() instead.
        Returns False otherwise.

        Returns:
            bool: Equality based on the rules described above
        """
        if other.__class__ == self.__class__:
            return self.identifiers() == other.identifiers()
        return NotImplemented

    def __eq__(self, o) -> bool:
        return self.matches(o)


def obj_to_codestr(o) -> str:
    """Returns a str representing a code object.

    If the object has a .codestr method, it will be used.
    If the object is an Enum member, str() will be used.
    If the object is a str-like object, it will be wrapped in quotes.
    If the object is a numeric, it will be returned as a string.
    """
    retval = None
    if o is None:
        return "None"
    elif hasattr(o, "codestr") and callable(o.codestr):
        retval = o.codestr()
    elif isinstance(o, Enum):
        retval = str(o)
    elif isinstance(o, str) or isinstance(o, AliasedName):
        retval = f"\"{str(o)}\""
    elif isinstance(o, Number):
        retval = str(o)
    elif isinstance(o, list):
        codestrs = map(obj_to_codestr, o)
        retval = "[" + ", ".join(codestrs) + "]"
    elif isinstance(o, dict):
        values = []
        for k, v in o.items():
            values.append((obj_to_codestr(k), obj_to_codestr(v)))
        values = [f"{t[0]}: {t[1]}" for t in values]
        retval = "{" + ", ".join(values) + "}"
    elif dataclasses.is_dataclass(o):
        class_name = o.__class__.__name__
        s = f"{class_name}("
        # Add each field as a new line
        L = []
        for f in dataclasses.fields(o):
            if not f.init:
                continue
            field_name = f.name
            field_str = obj_to_codestr(getattr(o, field_name))
            L.append(f"{field_name}={field_str}")
        s += ", ".join(L)
        s += ")"
        retval = s
    else:
        raise ValueError(f"Cannot convert object of type {type(o)} to code-string.")
    return retval
