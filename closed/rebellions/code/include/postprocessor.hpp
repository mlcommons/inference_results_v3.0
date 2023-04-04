#pragma once

#include <tvm/runtime/builtin_fp16.h>

#include <functional>
#include <type_traits>
#include <vector>

namespace rebel {

using PostProcessFuncType = std::function<void(const void* const*, void**)>;

void IdentityFunc(const void* const* result_from_model, void** output);

template <typename Tin, typename Tout, size_t LogitSize>
void ArgmaxFunc(const void* const* result_from_model, void** output) {
  Tout* max_index = reinterpret_cast<Tout*>(*output);

  auto result_from_model_typed = reinterpret_cast<const Tin* const*>(result_from_model);
  auto run = [&result_from_model_typed, &max_index](auto converter) {
    float max = -99999;
    for (Tout i = 0; i < LogitSize; i++) {
      auto value = converter(result_from_model_typed[0][i]);
      if (value > max) {
        max = value;
        *max_index = i;
      }
    }
  };

  if constexpr (std::is_same_v<Tin, uint16_t>) {
    constexpr float (*converter)(Tin) = &__gnu_h2f_ieee;
    run(converter);
  } else if constexpr (std::is_floating_point_v<Tin>) {
    constexpr float (*converter)(Tin) = [](Tin x) -> float { return x; };
    run(converter);
  } else
    throw std::exception();
}

class PostProcessor {
 public:
  virtual inline size_t GetSize() = 0;
  virtual const void* Get(const void* const* result_from_model) = 0;
};

class SimplePostProcessor : public PostProcessor {
 public:
  SimplePostProcessor(PostProcessFuncType func, size_t num_input, size_t out_bytes)
      : func_(func),
        num_input_(num_input),
        out_bytes_(out_bytes),
        container_(::operator new(out_bytes)) {}

  SimplePostProcessor(const SimplePostProcessor& others)
      : SimplePostProcessor(others.func_, others.num_input_, others.out_bytes_) {}

  virtual ~SimplePostProcessor() { ::operator delete(container_); }

  virtual inline size_t GetSize() { return out_bytes_; }

  const void* Get(const void* const* result_from_model) override {
    func_(result_from_model, &container_);
    return reinterpret_cast<void*>(container_);
  }

 private:
  PostProcessFuncType func_;
  size_t num_input_;
  size_t out_bytes_;
  void* container_;
};

class BatchPostProcessor : public PostProcessor {
 public:
  BatchPostProcessor(PostProcessFuncType func, size_t num_input, size_t batch_size, size_t in_bytes,
                     size_t out_bytes)
      : func_(func),
        num_input_(num_input),
        batch_size_(batch_size),
        in_bytes_(in_bytes),
        out_bytes_(out_bytes),
        container_(batch_size, nullptr) {
    for (auto& it : container_) it = malloc(out_bytes);
    input_cache_.resize(batch_size_);
    for (auto& it : input_cache_) it = new const void*[num_input_];
  }

  BatchPostProcessor(const BatchPostProcessor& others)
      : BatchPostProcessor(others.func_, others.num_input_, others.batch_size_, others.in_bytes_,
                           others.out_bytes_) {}

  virtual ~BatchPostProcessor() {
    for (auto& it : container_) free(it);
    for (auto& it : input_cache_) delete[] it;
  }

  virtual inline size_t GetSize() { return out_bytes_; }

  const void* Get(const void* const* result_from_model) override {
    auto result_from_model_casted = reinterpret_cast<const std::byte* const*>(result_from_model);
    for (size_t i = 0; i < num_input_; i++) {
      for (size_t j = 0; j < batch_size_; j++) {
        auto offset = j * in_bytes_;
        input_cache_[j][i] = result_from_model_casted[i] + offset;
      }
    }

    for (size_t i = 0; i < batch_size_; i++) func_(input_cache_[i], &(container_[i]));
    return container_.data();
  }

 private:
  PostProcessFuncType func_;
  size_t num_input_;
  size_t batch_size_;
  size_t in_bytes_;
  size_t out_bytes_;
  std::vector<const void**> input_cache_;
  std::vector<void*> container_;
};

}  // namespace rebel
