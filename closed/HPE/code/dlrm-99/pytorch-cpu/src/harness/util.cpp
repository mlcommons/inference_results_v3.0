#include <parallel/algorithm>
#include <cstdlib>
#include <numaif.h>
#include <glog/logging.h>
#include "kmp_launcher.hpp"
#include "util.hpp"

int mallocHost(void **ptr, size_t len) {
    // if (len % 64 != 0) {
    //     len = (len / 64 + 1) * 64;
    // }
    // *ptr = aligned_alloc(64, len);
    // if (*ptr == nullptr)
    //     return cpuFail;
    // return cpuSucc;
    constexpr int memalign = 4096;
    int rc = posix_memalign(ptr, memalign, len);
    if (rc == 0) {
        return cpuSucc;
    }
    return cpuFail;
}

int freeHost(void **ptr) {
    if (*ptr == nullptr)
        return cpuFail;
    free(*ptr);
    *ptr = nullptr;
    return cpuSucc;
}

std::vector<std::string>
splitString(const std::string& input,
            const std::string& delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while (next != std::string::npos) {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

// Restrict mem allocation to specific NUMA node.
void bindNumaMemPolicy(const int32_t numaIdx, const int32_t nbNumas) {
    unsigned long nodeMask = 1UL << numaIdx;
    long ret = set_mempolicy(MPOL_BIND, &nodeMask, nbNumas + 1);
    // CHECK(ret >= 0) << std::strerror(errno);
    // LOG(INFO) << "set_mempolicy: " << numaIdx << "/" << nbNumas << (ret ? " fail": " success");
}

// Reset mem allocation setting.
void resetNumaMemPolicy() {
    long ret = set_mempolicy(MPOL_DEFAULT, nullptr, 0);
    // CHECK(ret >= 0) << std::strerror(errno);
}

void bindThreadToCpus(int startIndex, int len) {
    kmp::KMPLauncher thCtrl;
    std::vector<int> places(len);
    for (int i = 0; i < len; ++i) {
        places[i] = i + startIndex;
    }
    thCtrl.setAffinityPlaces(places).pinThreads();
}

std::vector<double>
roc_auc_score(float *actual, float *prediction, int size, bool only_score) {
    std::vector<float> predictedRank(size, 0.0);
    int nPos = 0, nNeg = 0;
#pragma omp parallel for reduction(+:nPos)
    for(int i = 0; i < size; i++)
        nPos += (int)actual[i];

    nNeg = size - nPos;

    std::vector<std::pair<float, int> > v_sort(size);
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        v_sort[i] = std::make_pair(prediction[i], i);
    }

    __gnu_parallel::sort(v_sort.begin(), v_sort.end(), [](auto &left, auto &right) {
        return left.first < right.first;
    });

    int r = 1;
    int n = 1;
    size_t i = 0;
    while (i < size) {
        size_t j = i;
        while ((j < (v_sort.size() - 1)) && (v_sort[j].first == v_sort[j + 1].first)) {
            j++;
        }
        n = j - i + 1;
        for (size_t j = 0; j < n; ++j) {
            int idx = v_sort[i+j].second;
            predictedRank[idx] = r + ((n - 1) * 0.5);
        }
        r += n;
        i += n;
    }

    double filteredRankSum = 0;
#pragma omp parallel for reduction(+:filteredRankSum)
    for (size_t i = 0; i < size; ++i) {
        if (actual[i] == 1) {
            filteredRankSum += predictedRank[i];
        }
    }
    double score = (filteredRankSum - ((double)nPos * ((nPos + 1.0) / 2.0))) / ((double)nPos * nNeg);
    double log_loss = 0.0;
    double accuracy = 0.0;
    if (not only_score) {
        double acc = 0.0;
        double loss = 0.0;
#pragma omp parallel for reduction(+:acc,loss)
        for(int i = 0; i < size; i++) {
            auto rpred = std::roundf(prediction[i]);
            if(actual[i] == rpred) acc += 1;
            loss += (actual[i] * std::log(prediction[i])) +
                ((1 - actual[i]) * std::log(1 - prediction[i]));
        }
        accuracy = acc / size;
        log_loss = -loss / size;
    }

    return {score, log_loss, accuracy};
}
