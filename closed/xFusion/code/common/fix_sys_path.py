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

__doc__ = """Use this model to remove ~/.local/... Python import paths from sys.path. This is useful in the case where
developers have installed packages in their user module path which conflict with the ones installed in system paths.

By default, Python will prioritize site-packages in ~/.local over /usr/local/lib. This ordering cannot be changed by
PYTHONPATH, since it is a system installation default, so there are several solutions to this:

    1. Use a virtual_env and install all the dependencies in it, rather than the system. This requires the Makefile to
       activate this before running Python, as well as the developer to activate it for their shell in the case of
       debugging in a Python console or running scripts without using Makefile as an entrypoint.

       The other large issue with this is that *every line in Makefile executes in a separate shell*. This means every
       single invocation of Python must be prepended with `source path/to/venv/bin/activate &&`. This makes changing the
       Makefile very clunky and prone to errors.
    2. Use PYTHONHOME instead of a virtual_env, which contains every path in sys.path except ~/.local, so that we can
       effectively change the order of the import path. This makes it a one-time export in Makefile, but requires the
       developer to export it manually as an environment variable if they ever execute scripts without using Make.
    3. Directly edit sys.path and remove ~/.local site-packages from sys.path before importing. This can be done
       globally for the Python process, but can also be done on a per-file, as-needed basis by the programmer. Unlike
       (2), this also allows scripts to work when not executed through the Makefile.

This file implements the required functionality for method (3).
"""


import os
import sys


def _is_userhome_local(path):
    """Checks if a path resides in the ~/.local directory"""
    if not path.startswith(os.path.sep):
        return False

    path = os.path.expanduser(path)
    path_dirs = []
    while path != os.path.sep:
        path, head = os.path.split(path)
        path_dirs.append(head)
    return len(path_dirs) >= 4 and path_dirs[-1] == "home" and path_dirs[-3] == ".local"


# Remove duplicates and maintain ordering
_seen = set()
G_NO_USERHOME_LOCAL = tuple(x for x in sys.path if not _is_userhome_local(x) and not (x in _seen or _seen.add(x)))
del _seen


class ScopedRestrictedImport:
    """Scope where sys.path is temporarily modified. Any imports inside the scope will still be visible globally, but
    are imported from the restricted sys.path.

    After the scope ends, sys.path will be reverted back to its original value before the scope was created. This is NOT
    thread-safe.
    """

    def __init__(self, restricted_path=G_NO_USERHOME_LOCAL):
        """Creates a ScopedRestrictedImport where sys.path equals `restricted_path`. If no `restricted_path` is
        specified, defaults to `G_NO_USERHOME_LOCAL`.
        """
        self.old_sys_path = None
        self.restricted_path = restricted_path

    def __enter__(self):
        self.old_sys_path = sys.path[:]
        sys.path = self.restricted_path[:]
        return self

    def __exit__(self, type, value, traceback):
        sys.path = self.old_sys_path[:]

    def path_as_string(self):
        return ":".join(x for x in self.restricted_path if len(x) > 0)


def fix_pythonpath_command(cmd, pythonpath_extra=None):
    """Takes in a command in the form "python(3|3.*) *" and converts it to use an import path that does not include
    ~/.local.

    By default, Python will automatically import [site.py](https://docs.python.org/3/library/site.html) when the Python
    session is initialized. This is what inserts ~/.local/lib/python*/site-packages into sys.path automatically.
    However, this can by bypassed by using the -S flag. In doing so, we will need to add the system site-packages
    directories manually by generating a string and using PYTHONPATH.

    The reason this is not done within the Makefile is because it needs to be done at every invocation of Python within
    the Makefile, and also does not fix the problem with a subprocess is called (since the subprocess ALSO needs to pass
    in -S), and does not fix the issue when users run scripts by calling Python natively, without using Make.

    Args:
        cmd (str): The python command to convert. If the command does not start with 'python', throws an AssertionError
        pythonpath_extra (List[str]): A list of strings to add to the beginning

    Returns:
        str: The python command with a restricted PYTHONPATH explicitly exported.
    """
    assert cmd.startswith("python")
    toks = cmd.split(" ")
    # Get PYTHONPATH string
    with ScopedRestrictedImport() as sri:
        pythonpath = sri.path_as_string()
    if pythonpath_extra is not None:
        pythonpath_extra.append(pythonpath)
        pythonpath = ":".join(pythonpath_extra)
    toks.insert(1, "-S")
    return f"PYTHONPATH=\"{pythonpath}\" " + " ".join(toks)
