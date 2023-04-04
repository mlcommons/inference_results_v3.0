#ifndef LOADGEN_WRAPPER_CTRL_H_
#define LOADGEN_WRAPPER_CTRL_H_

#include <cstdint>
#include <condition_variable>
#include <future>
#include <mutex>

#include <loadgen.h>

struct LoadgenWrapperCtrl {
  // Mutex used to protect the access the data received from IssueQuery back from loadgen
  static std::mutex query_mutex;

  // Conditional variable to let loadgen know that data is available and ready from the laodgen
  static std::condition_variable query_cv;

  // Vector of responses back to loadgen library
  static std::vector<mlperf::ResponseId> next_example_ids;

  // The last seen variable used to keep track of last seen sample ids
  static size_t last_seen;

  static size_t next_example_ready;

  // Size of query samples ids returned by Loadgen
  static size_t next_example_size;

  // A copy of the next_exaple_ids used by the system to manipulate
  static std::vector<int64_t> cur_ids;

  // Atomic variable to indicate StartTest returned from loadgen
  static std::atomic<int> is_mlperf_done;

  // The future object returned by async StartTest thread
  static std::future<int> future;
};

#endif // LOADGEN_WRAPPER_CTRL_H_
