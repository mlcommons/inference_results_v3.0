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

#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <sola_runtime.h>

inline void Print(const char* text, int width) {
  printf("[=========");
  int len = strlen(text);
  if (len > width + 1) {
    width = len + 2;
  }
  int left = (width - len) / 2;
  int right = width - left - len;
  printf("%*s%s%*s", left, "", text, right, "");
  printf("=========]\n");
}

inline void PrintResult(int width, bool success) {
  printf("[=========");
  const char* text = "SAMPLE RESULT: Success";
  if (!success) {
    text = "SAMPLE RESULT: Failed ";
  }
  int len = strlen(text);
  if (len > width + 1) {
    width = len + 2;
  }
  int left = (width - len) / 2;
  int right = width - left - len;
  printf("%*sSAMPLE RESULT: ", left, "");
  if (success) {
    printf("\033[1;32mSuccess\033[0m");
  } else {
    printf("\033[1;31mFailed \033[0m");
  }
  printf("%*s=========]\n", right, "");
}

#define SampleBegin(name)   Print("RUN SAMPLE: " name, 40)
#define SampleEnd(name)     Print("END SAMPLE: " name, 40)
#define SampleSuccess()     PrintResult(40, true)
#define SampleFailed()      PrintResult(40, false)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    printf("[MFRT ERROR] at %s:%d code=%d(%s)\n\t\"%s\"\n", file, line,
           static_cast<unsigned int>(result), mfrtGetErrorName(result), func);
    SampleFailed();
    exit(EXIT_FAILURE);
  }
}
#define CheckError(val) check((val), #val, __FILE__, __LINE__)

inline bool CheckOutputCorrect(int* in, int* out, size_t size, const char* file, int line) {
  for (int i = 0; i < size; ++i) {
    if (in[i] != out[i]) {
      printf("[ERROR] %s:%d at:\n\tout[%d] = %d, in[%d] = %d\n", file, line, i, out[i], i, in[i]);
      return false;
    }
  }
  return true;
}
#define CheckCorrect(in, out, size) CheckOutputCorrect(in, out, size, __FILE__, __LINE__)

inline const char* GetModeName(MFMode mode) {
  switch (mode) {
    case MF_MODE_SINGLE_CORE_0: return "MF_MODE_SINGLE_CORE_0";
    case MF_MODE_SINGLE_CORE_1: return "MF_MODE_SINGLE_CORE_1";
    case MF_MODE_SINGLE_CORE_2: return "MF_MODE_SINGLE_CORE_2";
    case MF_MODE_SINGLE_CORE_3: return "MF_MODE_SINGLE_CORE_3";
    case MF_MODE_FOUR_CORE_BROADCAST: return "MF_MODE_FOUR_CORE_BROADCAST";
    case MF_MODE_FOUR_CORE_SPLIT: return "MF_MODE_FOUR_CORE_SPLIT";
    default: break;
  }
  return "unknown mode";
}

#endif // COMMON_COMMON_H_
