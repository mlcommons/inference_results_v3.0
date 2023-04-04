/*
 * Copyright (c) 2023, NEUCHIPS.  All rights reserved.
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
 *
*/

#ifndef NEUCHIPS_INFER_H

#define NEUCHIPS_INFER_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace ncsinfer1
{

class Dims32
{
public:
    static constexpr int32_t MAX_DIMS{8};
    int32_t nbDims;
    int32_t d[MAX_DIMS];
};
 
using Dims = Dims32;
 
enum class TensorFormat : int32_t
{
    kLINEAR = 0
};

enum class Severity : int32_t
{
    kINTERNAL_ERROR = 0,
    kERROR = 1,
    kWARNING = 2,
    kINFO = 3,
    kVERBOSE = 4,
};

}; // namespace ncsinfer1

#endif // NEUCHIPS_INFER_H
