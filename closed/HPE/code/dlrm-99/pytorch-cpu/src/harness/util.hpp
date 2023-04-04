#pragma once

#include <string>
#include <thread>
#include <vector>

constexpr int cpuSucc = 0;
constexpr int cpuFail = 1;
int mallocHost(void **ptr, size_t len);

int freeHost(void **ptr);

std::vector<std::string>
splitString(const std::string& input,
            const std::string& delimiter);

void bindNumaMemPolicy(const int32_t numaIdx, const int32_t nbNumas);

void resetNumaMemPolicy();

void bindThreadToCpus(int startIndex, int len);

std::vector<double>
roc_auc_score(float *actual, float *prediction, int size, bool only_score=true);
