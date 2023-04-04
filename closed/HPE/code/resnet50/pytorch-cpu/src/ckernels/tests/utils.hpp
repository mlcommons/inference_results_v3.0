#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <chrono>
#include <kernel_rn50/rn50_backbone.hpp>
#include <kernel_rn50/shape.hpp>
#include <omp.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <vector>

template <typename T, size_t N> size_t product(const std::array<T, N> &v) {
  size_t ret = 1;
  for (auto a : v) {
    ret *= a;
  }
  return ret;
}

template <typename T> struct distribution {};

template <> struct distribution<int8_t> {
  typedef std::uniform_int_distribution<int8_t> type;
};

template <> struct distribution<float> {
  typedef std::uniform_real_distribution<float> type;
};

template <typename T>
void uniform(const T min, const T max, size_t size, T *buf) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  typename distribution<T>::type distr(min, max);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    buf[i] = distr(generator);
  }
}

template <typename T, size_t alignment = 64> class aligned_vector {
public:
  aligned_vector(size_t size) {
    ptr_ = aligned_alloc(alignment, sizeof(T) * size);
  }
  aligned_vector(size_t size, T val) {
    ptr_ = aligned_alloc(alignment, sizeof(T) * size);
    for (size_t i = 0; i < size; i++) {
      static_cast<T *>(ptr_)[i] = val;
    }
  }
  ~aligned_vector() {
    if (ptr_)
      free(ptr_);
  }
  T *data() { return static_cast<T *>(ptr_); }
  aligned_vector(aligned_vector &&other) {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  aligned_vector &operator=(aligned_vector &&other) {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }

private:
  void *ptr_ = nullptr;
};

#endif