# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import itertools
from abc import ABC, abstractmethod
from collections import namedtuple
from math import log

Run = namedtuple("Run", ["cross", "stats"])


def do_send(gen, past=None):
    """ Functions as `next' for value-accepting generators.
    """
    try:
        cross = gen.send(past)
    except StopIteration:
        cross = None
    return cross


class SearcherInterface(ABC):
    @property
    @abstractmethod
    def is_dynamic(self):
        """ Whether or not this object's `generate' uses the return value of `yield'.

        Determines compatiblity with 'dry run' feature"""
    @abstractmethod
    def generate(self):
        """ A generator function which `yield's cross-terms. Invoked with `send', so can make use of past run info"""

    def get_best(self):
        """Given a list of runs, returns the 'best' config"""
        raise NotImplementedError("get_best not implemented")


class Step(SearcherInterface):
    """ Tries `search_param' from `start'->`end' (not including end) in steps of update(val). Repeated calls to update should converge exactly to `end'"""
    is_dynamic = False

    def __init__(self, search_param, start, end, update):
        self.search_param = search_param
        self.start = start
        self.end = end
        assert(self.start != self.end)
        self.update = update

    def generate(self):
        curr_val = self.start
        while curr_val != self.end:
            yield {self.search_param: curr_val}
            curr_val = self.update(curr_val)


class GreedyStepper(SearcherInterface):
    """Greedy search for a list of 'Step' """
    is_dynamic = True

    def __init__(self, steppers, predicate):
        self.steppers = steppers
        self.predicate = predicate
        self.best_cross = None
        for s in steppers:
            assert(isinstance(s, Step))

    def generate(self):
        # Start with initial terms:
        gens = []
        cross = {}
        # Reversing isn't strictly necessary, but is nice because "pop()" pops from the back, so we get expected order
        for s in reversed(self.steppers):
            gen = s.generate()
            cross.update(do_send(gen))
            gens.append(gen)
        baseline = yield cross
        self.best_cross = cross.copy()
        gen = None
        while len(gens):
            if gen is None:
                gen = gens.pop()
            partial = do_send(gen)
            if partial:
                # Valid term
                cross.update(partial)
                result = yield cross
                if self.predicate(baseline, result):
                    baseline = result
                    self.best_cross = cross.copy()
                    continue
                else:
                    cross = self.best_cross.copy()
            # Reset because we didn't like the result:
            gen = None

    def get_best(self):
        return self.best_cross.copy()


class CartesianProduct(SearcherInterface):
    """A 'dumb' search, which enumerates all terms in `grid', optionally ordered to reduce the number of engine rebuilds"""
    is_dynamic = False

    def __init__(self, grid, no_rebuild_params=None):
        self.grid = grid
        self.no_rebuild_params = no_rebuild_params

    def generate(self):
        # We can be fancy:
        # If we choose the noRebuildNeeded params to be on the innerDim of the cross product, that will minimize the number of rebuilds we need
        # (Because we only have the last engine cached).
        # Note, in the general case (of scheduling arbitrary jobs where the only thing known is if running job B after job A requires a rebuild), this reduces to finding a minimum cost hamiltonian path, which is NP Complete
        if self.no_rebuild_params:
            sorted_keys = sorted(self.grid.keys(), key=lambda x: x in self.no_rebuild_params)
            sorted_vals = (self.grid[k] for k in sorted_keys)
            cross_terms = (i for i in itertools.product(*sorted_vals))
            named_terms = (dict(zip(sorted_keys, term)) for term in cross_terms)
        else:
            cross_terms = (i for i in itertools.product(*self.grid.values()))
            named_terms = (dict(zip(self.grid.keys(), term)) for term in cross_terms)
        for term_dict in named_terms:
            yield term_dict


class Bisect(SearcherInterface):
    """ Runs binary search for search_param.

    Looks over the range [`lower_bound', `upper_bound') for the rightmost value for which predicate returns True.
    `predicate' takes a `Run' object and returns True or False.
    """
    is_dynamic = True

    def __init__(self, search_param, lower_bound, upper_bound, step_size, predicate):
        self.search_param = search_param
        self.plausible = range(lower_bound, upper_bound, step_size)
        assert(len(self.plausible) > 2)
        self.step_size = step_size
        self.predicate = predicate
        self.best = {self.search_param: lower_bound}

    def was_good(self, past_run):
        return self.predicate(past_run.stats)

    def generate(self):
        lower_idx = 0
        upper_idx = len(self.plausible)
        while lower_idx + 1 != upper_idx:
            mid_idx = (upper_idx + lower_idx) // 2
            term = {self.search_param: self.plausible[mid_idx]}
            # The following statement both
            # - Returns the term to the caller
            # - Resumes execution with `result' taking the value of the caller-sent-object ("Run").
            result = yield term
            if self.predicate(result):
                lower_idx = mid_idx
                self.best = term.copy()
            else:
                upper_idx = mid_idx

    def get_best(self):
        return self.best


class FindUpperBound(SearcherInterface):
    """ Given a starting point, will attempt to maximize the parameter efficiently.

    An example of 'delegating' to another searcher internally via 'yield from'"""
    is_dynamic = True

    def __init__(self, search_param, start, predicate, num_bisect_steps):
        self.search_param = search_param
        self.start = start
        self.predicate = predicate
        self.num_bisect_steps = num_bisect_steps
        self.best = {self.search_param: start}

    def generate(self):
        # First we need to find an upper bound, we can do this by jumping in powers of two:
        curr_value = self.start * 2
        result = yield {self.search_param: curr_value}
        while self.predicate(result):
            curr_value *= 2
            result = yield {self.search_param: curr_value}
        upper_bound = int(curr_value)
        lower_bound = int(curr_value / 2)
        # If we want to do at most num_bisect_steps, we know that the number of elements we will need to check is (upper_bound - lower_bound)/StepSize, so the number of checks done by binary search is log_2(Range/StepSize) = num_steps, so
        # Range / 2^(num_steps) = step_size
        step_size = int((upper_bound - lower_bound) / (2**self.num_bisect_steps))
        bisector = Bisect(search_param=self.search_param,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          step_size=step_size,
                          predicate=self.predicate)
        g = bisector.generate()
        yield from g
        self.best = bisector.get_best()

    def get_best(self):
        return self.best.copy()


class Overlay(SearcherInterface):
    """ Overlays `overlay_dict' to each cross term produced by `subject' """
    @property
    def is_dynamic(self):
        return self.subject.is_dynamic

    def __init__(self, subject, overlay_dict):
        self.subject = subject
        self.overlay_dict = overlay_dict

    def generate(self):
        gen = self.subject.generate()
        term_from_gen = do_send(gen)
        while term_from_gen:
            cross = term_from_gen
            cross.update(self.overlay_dict)
            result = yield cross
            term_from_gen = do_send(gen, result)

    def get_best(self):
        subject_best = self.subject.get_best()
        subject_best.update(self.overlay_dict)
        return subject_best


class Composer(SearcherInterface):
    """ Compose multiple SearcherInterfaces with customizable logic to connect them.

    Each SearcherInterface must be associated with a predicate which returns True if we should use the next item
    from the generator, and False if we should query the next generator.
    This predicate should take a single `Run' and return True or False. This predicate can be (in preference order):
    - Passed in as the second position of a tuple [ie: (SearcherInterfaceObject, predicate)]
    - Used as the same `predicate' property of the SearcherInterface in question
    - Always True (Default) """
    is_dynamic = True

    def __init__(self, searchersAndPreds, timeout):
        self.timeout = timeout
        self.searchers = []
        self.predicates = []
        def default_predicate(x): return True

        self.timeout = timeout
        for item in searchersAndPreds:
            if isinstance(item, tuple):
                # Expect (Searcher, lambda)
                assert(len(item) == 2)
                assert(callable(item[1]))
                self.searchers.append(item[0])
                self.predicates.append(item[1])
            else:
                self.searchers.append(item)
                # if item has a predicate, we capture that, otherwise, use our default:
                if hasattr(item, "predicate"):
                    assert(callable(item.predicate))
                    self.predicates.append(item.predicate)
                else:
                    self.predicates.append(default_predicate)

    def generate(self):
        curr_cross = {}
        gens = []
        num_tries = 0
        # Walk through all searchers to get initial values:
        for s in self.searchers:
            gen = s.generate()
            curr_cross.update(do_send(gen))
            gens.append(gen)

        while num_tries != self.timeout:
            past_run = yield curr_cross
            num_tries += 1
            past_cross = curr_cross.copy()
            term_to_twiddle = None
            for g, pred in zip(gens, self.predicates):
                if pred(past_run):
                    term_to_twiddle = do_send(g, past_run)
                    # May be None, in which case, we need to query the next item
                    if term_to_twiddle:
                        break
            if term_to_twiddle is None:
                # If we walked through all our generators, and nothing wanted to run, we "forcefully"
                # query all generators (but backwards/LIFO style)
                for g in reversed(gens):
                    term_to_twiddle = do_send(g, past_run)
                    if term_to_twiddle:
                        break
                if term_to_twiddle is None:
                    # If we _still_ have nothing to send, we're completely exausted, so we're done generating
                    break
            # We now have a partial term:
            curr_cross.update(term_to_twiddle)
