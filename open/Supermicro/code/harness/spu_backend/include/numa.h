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

#ifndef SPU_BACKEND_NUMA_H_
#define SPU_BACKEND_NUMA_H_

#include <iostream>
#include <vector>
#include <pthread.h>
#include <memory>

namespace moffett {
namespace spu_backend {

void* MallocHostMemory(size_t size, int numa);
int FreeHostMemory(void* vaddr, size_t size, int numa);

}
}

#endif // SPU_BACKEND_NUMA_H_
