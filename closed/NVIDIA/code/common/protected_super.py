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

import inspect


__doc__ = """Use this module for some weird cases of Python inheritance. In some cases, it is possible for super() to
not work as desired, especially with nested function calls when some methods are not overridden by child classes.

For example:

```
class A:
    def f1(self, x):
        print("A.f1", x)

    def f2(self, x):
        print("A.f2", x)

    def run(self, x):
        self.f1(x)
        self.f2(x)

class B(A):
    def f1(self, x):
        print("B.f1", x)

    def f2(self, x):
        print("B.f2", x)
        super().run()

B().run("hi")
```

In this case, the intent is that it will print:
>>> B.f1 hi
    B.f2 hi
    A.f1 hi
    A.f2 hi

However, in practice, because A.run calls `self.<method>`, the use of `super()` does something unintended where the
`self.f1` and `self.f2` calls in `A.run` will use the f1 and f2 overridden in B, which then causes an infinite recursion
(B.f2 calls run again).
"""


class ProtectedSuper:
    """Creates a scope where an object is interpreted as its parents class.

    Example usage:

        class A:
            def f1(self, x):
                print("A.f1", x)

            def f2(self, x):
                print("A.f2", x)

            def run(self, x):
                self.f1(x)
                self.f2(x)

        class B(A):
            def f1(self, x):
                print("B.f1", x)

            def f2(self, x):
                print("B.f2", x)
                with ProtectedSuper(self) as duper:
                    duper.run(x)

        class C(B):
            def f1(self, x):
                print("C.f1", x)

            def f2(self, x):
                print("C.f2", x)
                with ProtectedSuper(self) as duper:
                    # self will still be an instance of `C`
                    self.f1(x + " start")  # ProtectedSuper does not modify 'self'! Can still use as normal.
                    duper.run(x)  # duper is an instance of `B`
                    self.f1(x + " end")

        c = C()
        c.run("hi")

    Here, this will call (in order):
        1. C.f1("hi")
        2. C.f2("hi")
            1. C.f1("hi start")
            2. B.run("hi")
                1. B.f1("hi")
                2. B.f2("hi")
                    1. A.f1("hi")
                    2. A.f2("hi")
            3. C.f1("hi end")
    """

    def __init__(self, obj):
        """Creates a ProtectedSuper object for a given object instance. The super object will use the immediate parent
        class of `obj` from its Method Resolution Order.
        """
        self.__base_instance__ = obj
        self.__original_cls__ = obj.__class__
        parents = inspect.getmro(obj.__class__)
        self.__immediate_parent_cls__ = parents[1]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__base_instance__.__class__ = object.__getattribute__(self, "__original_cls__")

    def __getattr__(self, name):
        r = None
        self.__base_instance__.__class__ = object.__getattribute__(self, "__immediate_parent_cls__")
        if hasattr(self.__base_instance__, name):
            r = getattr(self.__base_instance__, name)
        self.__base_instance__.__class__ = object.__getattribute__(self, "__original_cls__")

        # Handle cleanup. If r is a callable, we need to call under superclass, and reset after the call completes.
        if callable(r):
            def _f(*args, **kwargs):
                try:
                    self.__base_instance__.__class__ = object.__getattribute__(self, "__immediate_parent_cls__")
                    return r(*args, **kwargs)
                finally:
                    self.__base_instance__.__class__ = object.__getattribute__(self, "__original_cls__")
            return _f
        return r
