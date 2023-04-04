It would be nice to have an automated, scenario/benchmark agnostic means of getting timings for a variety of parameters.

Immediately, this is useful for knob-heavy benchmarks like RNN-T, but will also be useful in the future as MLPerf becomes more system-focused (thereby increasing the number of potential parameters needing to be tuned).

[[_TOC_]]


# Usage

Requires that the given scenario, benchmark, and system have a present baseline configuration in `configs/` (the typical usage pattern for a new system, is to copy-and-paste an already-tuned system's config entry, and use the autotuner to find a more suitable parameter set).

See

    scripts/autotune/grid.py --help

for arguments. Run the script with `make autotune RUN_ARGS="..."`.

For "simple" jobs (run all possible combinations of all parameters), it is suggested to use JSON input.

For more complex jobs (where certain combinations shouldn't be executed, special properties are known about the parameters to optimize scheduling, etc), it is suggested to use Python input.
Refer to [Config Schema](#Config-Schema) for additional information.


# Config Schema


## JSON

As seen in `scripts/autotune/example.json`, the expected format is a flat object whose properties are parameter names, and whose values are arrays of non-object, non-array items. Practically, this is bools, numbers, or strings; identical to the python case.


## Python

As seen in `scripts/autotune/example.py`, the expected format is variables with no leading underscores declared at the global namespace which are arrays of primitive, non-dict, non-array type. Practically this is bools, int, float, and str; identical to the JSON case.

Special functions prefixed by `META_` define special hooks into the scheduling process. They are described below:



### META_get_no_rebuild_params()

A function which takes no arguments and returns a list of strings. These strings MUST correspond to variable names in the same Python file (and these strings must also therefore describe valid parameters for the given benchmark/scenario) whose values can be changed by the autotuner without rebuilding the engine.

The scheduler will use the variable names returned by this function to order runs in such a manner to reduce the number of rebuilds (`make generate_engines`) required to fully execute the job.

#### Example

Take the following pathological case:

    audio_batch_size = [128,256,512] # A runtime parameter (doesn't require rebuilding engines)
    gpu_batch_size = [128,256] # A buildtime AND runtime parameter (requires rebuilding engines)

A naive scheduling of these parameters could produce

    audio_batch_size = 128; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 128; gpu_batch_size = 256; 
    # Rebuild!
    audio_batch_size = 256; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 256; gpu_batch_size = 256; 
    # Rebuild!
    audio_batch_size = 512; gpu_batch_size = 128; 
    # Rebuild
    audio_batch_size = 512; gpu_batch_size = 256; 

Requiring 5 rebuilds because a build-time parameter changes at each step

If we defined

    def META_get_no_rebuild_params():
        return ['audio_batch_size']

Our scheduler can produce the following intelligent ordering:

    audio_batch_size = 128; gpu_batch_size = 128;
    audio_batch_size = 256; gpu_batch_size = 128;
    audio_batch_size = 512; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 128; gpu_batch_size = 256;
    audio_batch_size = 256; gpu_batch_size = 256;
    audio_batch_size = 512; gpu_batch_size = 256;

Doing only one rebuild instead! (Potentially reducing our tuning time by an equal order of magnitude for some benchmarks)



### META_is_config_valid()

A callback which takes a "full parameter config dict" (the default config from `configs/benchmark/scenario` updated/overlayed with values from the configuration file), and returns whether or not this configuration is valid.

If this callback is not defined, the default behavior is to treat all configs as valid

#### Example

Take the following parameter lists:

    audio_batch_size = [128, 256, 512, 1024]
    dali_pipeline_depth = [1,2,3]
    audio_buffer_num_lines = [128, 256, 512, 1024, 2048, 4096]

And we know that due to the nature of our benchmark, if `audio_batch_size * dali_pipeline_depth` is ever greater than `audio_buffer_num_lines`, our benchmark will crash.
So, the following is true:

    audio_batch_size = 256; dali_pipeline_depth=1; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 512; dali_pipeline_depth=1; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 512; dali_pipeline_depth=2; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 1024; dali_pipeline_depth=2; audio_buffer_num_lines = 1024 # CRASH!

Instead of manually specifying all legal configurations (the exact problem that autotuning solves). What would be nice is if the autotuner could just not run certain configurations based on a predicate. As it turns out, the autotuner script has this interface! It's presents a callback which states if is a configuration is legal before launching work (`generate_engines` and `run_harness`).

So, we can specify:

```python
def META_is_config_valid(config):
    if config['dali_pipeline_depth'] * config['audio_batch_size'] > config['audio_buffer_num_lines']:
        return False
    return True
```

And now the failing cases will not be executed.

### META_search_callback()

A function which takes no arguments and returns an object which implements `SearcherInterface`. Examples of concrete implementations include `Step`,  `CartesianProduct` (default search behavior), `Bisect`, and `Composer`.

This should be used when the default behavior of exhaustively iterating through all parameter combinations is not desired.

#### `SearcherInterface`

An interface containing an abstract method `generate()` and an abstract property `is_dynamic`. `generate()` is a [generator function](https://docs.python.org/3/glossary.html#term-generator), which can optionally accept incremental results in the form of `Run` (see `Bisect::generate()` for client usage, see [send](https://docs.python.org/3/reference/expressions.html?#generator.send) for language documentation) to influence searching. `is_dynamic` should be set to `True` if `generate()` uses incremental results.


#### Helpers

Refer to `library.py` which provides many utilities. For example, `Step` can be imported with `from library import Step` (note that package lookup is relative to `grid.py`, not the configuration py file, so no fancy relative pathing or `sys.path.insert(foo)` needed.)

##### Example: Binary Search with `Bisect`

Suppose we want to perform a binary search to find an optimal QPS value given default parameters in the QPS range: `(500, 2500)` in step-sizes of 100. We can use `Bisect` to make the following

```python
from library import Bisect

def META_search_callback(past_runs):
    return Bisect(search_param="server_target_qps",
                  lower_bound=3000,
                  upper_bound=4650,
                  step_size=50,
                  predicate=lambda r: r.stats['result_validity'] == "VALID"))

# We know QPS is a runtime-only parameter, so we don't need to rebuild in between runs:
def META_get_no_rebuild_params():
    return ["server_target_qps"]
```

##### `Composer`

As a reference implementation of a `SearcherInterface` which non-trivially utilizes incremental results, `Composer` is a searcher which composes multiple SearcherInterface objects composed by optional predicates.

The logic which `Composer` takes is as follows:
 1. Ask all searchers for their initial search term and run inference
 2. Ask the first searcher if it "wants" to yield its next object given the past run
 3. If it does, ask the first searcher for its next search term
 4. If not, repeat #2-3 for all searchers until we find one that wants to give us a search term, and ask for its next search term
 5. If no searchers "want" to yield the next search term, force all searchers, backwards, to give us a term. We stop querying once a searcher gives us a term
 6. If no searchers gave us a term when forced, we're done.

Note that we separate "asking what a searcher _wants_ to do" and "asking a searcher for its next search term" because calling `next`/`send` is destructive, and it's quite a headache to manually rebuild state after each query to a generator.

Refer to the following diagram for a pictoral representation of the aforementioned logic.

```plantuml
  start
  !pragma useVerticalIf on
  while (timeout?)
      :Run and acquire RunArtifact;
      if (predicate1(RunArtifact)) then (True)
          :search_term = get(Searcher1) ;
      elseif (predicate2(RunArtifact)) then (True)
          :search_term = get(Searcher2) ;
      endif
      if (search_term == NULL) then (True)
          :Normal consultation has failed, now force each searcher to give us a term
          search_term = get(Searcher2) || get(Searcher1);
      endif
      if (search_term == NULL) then (True)
          :We searched through everything
          break!;
      endif
      :++NumRuns;
  endwhile (Yes)
  :Finish;
  end
```

##### Example: Somewhat intelligent server tuning. With and without Composer

One such application, would be to tune `server_target_qps` at a top level using `Bisect` , but also schmoo other knobs such as `gpu_inference_streams` and `gpu_batch_size` before we take a negative step in our binary search. This can be implemented as follows:

```python
def META_search_callback():
    return Composer([Bisect(search_param="server_target_qps",
                            lower_bound=3000,
                            upper_bound=4650,
                            step_size=50,
                            predicate=lambda r: r.stats['result_validity'] == "VALID"), # Composer will use Bisect's predicate
                     (Step(search_param="gpu_inference_streams",
                           start=1,
                           end=4,
                           update=lambda x: x + 1),
                      lambda x: True), # Redundant, but demonstrates how we can have an arbitrary predicate for Composer's logic
                     Step(search_param="gpu_batch_size",
                          start=200,
                          end=100,
                          update=lambda x: x - 50)], # By default, we always return True.
                timeout=10)
```

If one were to manually implement this logic as a bespoke SearcherInterface, it would look something like:

```python
class Bespoke(SearcherInterface):
    is_dynamic = True
    def __init__(self, timeout=10):
        self.timeout=timeout

    def generate(self):
        b = Bisect(search_param="server_target_qps",
                   lower_bound=3000,
                   upper_bound=4650,
                   step_size=50,
                   predicate=lambda p: p.stats['result_validity'] == "VALID")

        qps_gen = b.generate()
        is_gen = Step(search_param="gpu_inference_streams",
                      start=1,
                      end=4,
                      update=lambda x: x + 1).generate()
        bs_gen = Step(search_param="gpu_batch_size",
                      start=200,
                      end=100,
                      update=lambda x: x - 50).generate()
        curr_cross = {}
        # Generate first run cross:
        curr_cross = do_send(qps_gen)
        curr_cross.update(do_send(is_gen))
        curr_cross.update(do_send(bs_gen))
        num_runs=0
        while num_runs != self.timeout:
            past_run = yield curr_cross
            num_runs += 1
            past_cross = curr_cross.copy()
            if b.was_good(past_run):
                # First run done, we can increase server_target_qps
                partial = do_send(qps_gen, past_run)
            else:
                # We want to decrease server_target_qps, but let's try other stuff first:
                partial = do_send(is_gen)
                if partial is None:
                    # That was exhausted, let's try another:
                    partial = do_send(bs_gen)
                    if partial is None:
                        # We must do the qps reduction:
                        partial = do_send(qps_gen, past_run)
            if partial is None:
                # If either branch of qps_gen gave us None, catch it here:
                break
            else:
                curr_cross.update(partial)
```


