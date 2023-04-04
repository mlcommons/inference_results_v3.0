# NVIDIA MLPerf Inference Custom Systems

NVIDIA's MLPerf Inference submission uses a Python-based object to describe various system specifications and
descriptions. The objects for this are contained in the `code.common.systems` module.


As this format can be complicated for the inexperienced user, NVIDIA provides a codepath and script to automatically
detect the current system's specifications and add it to the codebase. The script will set up the system detection and
provide stubs for the Benchmark configurations.

## Running the script

The script is located at `scripts/custom_systems/add_custom_system.py`. Run the script either directly or via Python
inside the Docker container:

```
$ ./scripts/custom_systems/add_custom_system.py

# OR 

$ python3 scripts/custom_systems/add_custom_system.py
```

Running this script will provide you some details about the running system and go through a variety of prompts to set up
the stubs and system description.
