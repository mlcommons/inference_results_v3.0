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

#include "multithreading.h"

namespace moffett {
namespace spu_backend {

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Create thread
MFThread mfStartThread(MF_THREADROUTINE func, void *data) {
  return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

// Wait for thread to finish
void mfEndThread(MFThread thread) {
  WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
}

// Wait for multiple threads
void mfWaitForThreads(const MFThread *threads, int num) {
  WaitForMultipleObjects(num, threads, true, INFINITE);

  for (int i = 0; i < num; i++) {
    CloseHandle(threads[i]);
  }
}

#else
// Create thread
MFThread mfStartThread(MF_THREADROUTINE func, void *data) {
  pthread_t thread;
  pthread_create(&thread, NULL, func, data);
//    pthread_detach(thread);
  return thread;
}

MFThread mfDetachThread(MF_THREADROUTINE func, void *data) {
  pthread_t thread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_create(&thread, &attr, func, data);
  return thread;
}

// Wait for thread to finish
void mfEndThread(MFThread thread) { pthread_join(thread, nullptr); }

// Wait for multiple threads
void mfWaitForThreads(const MFThread *threads, int num) {
  for (int i = 0; i < num; i++) {
    mfEndThread(threads[i]);
  }
}

// Increment barrier. (execution continues)
void MFBarrier::Increment() {
  std::lock_guard<std::mutex> guard(mutex_);
  ++count_;

  if (count_ >= release_count_) {
    con_var_.notify_all();
  }
}

// Wait for barrier release.
void MFBarrier::WaitForRelease() {
  std::unique_lock<std::mutex> guard(mutex_);
  if (count_ < release_count_) {
    con_var_.wait(guard, [this] {
      return count_ >= release_count_;
    });
  }
}

#endif

}
}
