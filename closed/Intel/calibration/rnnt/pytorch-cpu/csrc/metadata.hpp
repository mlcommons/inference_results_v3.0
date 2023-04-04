#pragma once
#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>
#include <math.h>
#include <query_sample_library.h>
#include <torch/script.h>

#include <string>

namespace rnnt {
using TensorVector = std::vector<at::Tensor>;
using QuerySample = mlperf::QuerySample;
using namespace torch::indexing;

static const std::vector<char> LABELS = {
    ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};

enum Params {
  PRE_NUM_LAYERS = 2,
  POST_NUM_LAYERS = 3,
  TRANS_INPUT_SIZE = 240,
  TRANS_HIDDEN_SIZE = 1024,
  STACK_TIME_FACTOR = 2,
  PRED_NUM_LAYERS = 2,
  PRED_HIDDEN_SIZE = 320,
  JOINT_HIDDEN_SIZE = 512,
  SOS = -1,
  BLANK = 28,
  MAX_SYMBOLS_PER_STEP = 30,
  MAX_WAV_LEN = 240000,
  MAX_FEA_LEN = 500,
  PADDED_INPUT_SIZE = 256
};

// for Offline: batch ahead
class State {
public:
  State(){};
  State(int32_t batch_size, int32_t split_len = -1) {
    init(batch_size, split_len);
    auto intra = at::get_num_threads();
    padded_batch_size_ = ceil(batch_size_ / (intra * 16)) * (intra * 16);
  };
  virtual ~State() = default;
  void init(int32_t batch_size, int32_t split_len = -1);
  void update(
      at::Tensor x, at::Tensor x_lens, int32_t split_len = -1,
      int32_t actual_batch_size_ = -1);
  bool next();
  void clear();

  int32_t finish_size_;
  int32_t actual_batch_size_;
  int32_t padded_batch_size_;
  int32_t batch_size_;
  int32_t split_len_;
  int32_t padded_fea_len_ = MAX_FEA_LEN;
  int32_t max_res_len_ = padded_fea_len_ / 2 * MAX_SYMBOLS_PER_STEP;
  at::Tensor split_lens_;
  // transcription
  TensorVector f_split_;
  at::Tensor f_;
  at::Tensor f_lens_;
  TensorVector pre_hx_;
  TensorVector pre_cx_;
  TensorVector post_hx_;
  TensorVector post_cx_;
  // prediction
  at::Tensor pre_g_;
  TensorVector pre_hg_;
  TensorVector pre_cg_;
  // results
  at::Tensor res_;
  at::Tensor res_idx_;
  // infer index
  at::Tensor finish_idx_;
  at::Tensor remain_lens_;
  at::Tensor infer_lens_;
  int32_t split_idx_ = 0;
};

// for Server: batch dynamic
class PipelineState : public State {
public:
  PipelineState();
  PipelineState(int32_t batch_size, int32_t split_len, int32_t response_size)
      : finish_size_(batch_size), response_size_(response_size) {
    init(batch_size, split_len);
    auto intra = at::get_num_threads();
    padded_batch_size_ = ceil(batch_size_ / (intra * 16)) * (intra * 16);
  };
  virtual ~PipelineState() = default;
  void init(int32_t batch_size, int32_t split_len = -1);
  void update(
      std::vector<std::tuple<QuerySample, at::Tensor, at::Tensor>>
          &dequeue_list,
      std::vector<QuerySample> &samples, int32_t dequeue_size,
      int32_t split_len);
  bool next();

  int32_t finish_size_;    // = require_size = padd_size_ + actual_response_size
  int32_t response_size_;  // actual_response_size >= response_size
  int32_t stop_size_;      // = min(batch_size_, padded_size_ + response_size_)
  int32_t remain_size_ = 0;   // remain_size_' = batch_size_ - finish_size_
  int32_t dequeue_size_ = 0;  // <= finish_size_
  int32_t padded_size_ = 0;  // = batch_size_ - remain_size_ - dequeue_size_
  // transcription
  at::Tensor F_;
  at::Tensor F_lens_;
  // infer index
  at::Tensor split_idx_;
  at::Tensor eos_idx_;
};

}  // namespace rnnt
