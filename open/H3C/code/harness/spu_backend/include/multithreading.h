/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPU_BACKEND_MULTITHREADING_H
#define SPU_BACKEND_MULTITHREADING_H

// POSIX threads.
#include <pthread.h>
#include <mutex>
#include <condition_variable>

namespace moffett {
namespace spu_backend {

typedef pthread_t MFThread;
typedef void *(*MF_THREADROUTINE)(void *);

#define MF_THREADPROC void *
#define MF_THREADEND return 0

struct MFBarrier {
 public:
  MFBarrier() = default;
  void SetReleaseCount(int release_count) {
    release_count_ = release_count;
  }
  // Increment barrier.
  void Increment();

  // Wait for barrier release.
  void WaitForRelease();

 private:
  std::mutex mutex_;
  std::condition_variable con_var_;
  int release_count_ = 0;
  int count_ = 0;
};

// declaration

#ifdef __cplusplus
extern "C" {
#endif

// Create thread.
MFThread mfStartThread(MF_THREADROUTINE, void *data);

// Detach thread.
MFThread mfDetachThread(MF_THREADROUTINE, void *data);

// Wait for thread to finish.
void mfEndThread(MFThread thread);

// Wait for multiple threads.
void mfWaitForThreads(const MFThread *threads, int num);

#ifdef __cplusplus
} //extern "C"
#endif

}
}

#endif //SPU_BACKEND_MULTITHREADING_H
