#include "metadata.hpp"

namespace rnnt {

void State::init(int32_t batch_size, int32_t split_len) {
  clear();
  batch_size_ = batch_size;
  split_len_ = split_len;
  if (split_len_ > 0)
    split_lens_ = at::full({batch_size_}, split_len, torch::kInt32);
  // init transcription tensors
  for (int32_t layer = 0; layer < PRE_NUM_LAYERS; ++layer) {
    pre_hx_.emplace_back(
        torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kInt8));
    pre_cx_.emplace_back(
        torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kFloat16));
  }
  for (int32_t layer = 0; layer < POST_NUM_LAYERS; ++layer) {
    post_hx_.emplace_back(
        torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kInt8));
    post_cx_.emplace_back(
        torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kFloat16));
  }
  // init prediction tensors
  pre_g_ = torch::full({1, batch_size_}, SOS, torch::kInt32);
  for (int64_t layer = 0; layer < PRED_NUM_LAYERS; ++layer) {
    pre_hg_.emplace_back(torch::empty(
        {batch_size_, PRED_HIDDEN_SIZE}, at::ScalarType::BFloat16));
    pre_cg_.emplace_back(
        torch::empty({batch_size_, PRED_HIDDEN_SIZE}, torch::kFloat32));
  }
  // init res tensors
  res_ = torch::empty({batch_size_, max_res_len_}, torch::kInt32);
  res_idx_ = torch::empty({batch_size_}, torch::kInt32);
}

void State::update(
    at::Tensor f, at::Tensor f_lens, int32_t split_len,
    int32_t actual_batch_size) {
  actual_batch_size_ = actual_batch_size;
  auto padded_batch_size = f_lens.size(0);
  if (padded_batch_size != batch_size_) init(padded_batch_size, split_len);
  // update transcription tensors
  for (int32_t layer = 0; layer < PRE_NUM_LAYERS; ++layer) {
    pre_hx_[layer].zero_();
    pre_cx_[layer].zero_();
  }
  for (int32_t layer = 0; layer < POST_NUM_LAYERS; ++layer) {
    post_hx_[layer].zero_();
    post_cx_[layer].zero_();
  }
  // update prediction tensors
  pre_g_.fill_(SOS);
  for (int64_t layer = 0; layer < PRED_NUM_LAYERS; ++layer) {
    pre_hg_[layer].zero_();
    pre_cg_[layer].zero_();
  }
  // update res tensors
  res_.fill_(SOS);
  res_idx_.fill_(-1);
  // update f & f_lens
  if (split_len_ > 0) {
    f_split_ = torch::split(f, split_len_);
    infer_lens_ = f_lens;
    remain_lens_ = f_lens.clone();
    split_idx_ = 0;
    finish_size_ = 0;
  } else {
    f_ = f;
    f_lens_ = f_lens;
    infer_lens_ = f_lens;
    finish_size_ = batch_size_;
  }
}

bool State::next() {
  bool status = (finish_size_ != batch_size_);
  if (status) {
    f_lens_ = torch::min(split_lens_, remain_lens_);
    f_ = f_split_[split_idx_++];
    remain_lens_ -= f_lens_;
    finish_idx_ = remain_lens_.le(0);
    finish_size_ = finish_idx_.count_nonzero().item().toInt();
  }
  return status;
}

void State::clear() {
  pre_hx_.clear();
  pre_cx_.clear();
  post_hx_.clear();
  post_cx_.clear();
  pre_hg_.clear();
  pre_cg_.clear();
}

void PipelineState::init(int32_t batch_size, int32_t split_len) {
  // ensure T can be evenly divided by split_len
  if (split_len > 0) {
    padded_fea_len_ = (MAX_FEA_LEN + split_len - 1) / split_len * split_len;
    max_res_len_ = padded_fea_len_ / 2 * MAX_SYMBOLS_PER_STEP;
  }
  State::init(batch_size, split_len);
  F_ = torch::empty({padded_fea_len_, batch_size_, PADDED_INPUT_SIZE});
  F_lens_ = torch::empty({batch_size_}, torch::kInt32);
  infer_lens_ = torch::empty({batch_size_}, torch::kInt32);
  remain_lens_ = torch::zeros({batch_size_}, torch::kInt32);
  finish_idx_ = torch::full({batch_size_}, true, torch::kBool);
}

void PipelineState::update(
    std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>>
        &dequeue_list,
    std::vector<mlperf::QuerySample> &samples, int32_t dequeue_size,
    int32_t split_len) {
  dequeue_size_ = dequeue_size;
  // currently use fixed batch_size & split_len!
  // actual_batch_size_ = dequeue_size_ + remain_size_;
  // if (actual_batch_size_ > batch_size_ || split_len != split_len_)
  //   PipelineState::init(actual_batch_size_, split_len);
  // update transcription tensors
  if (dequeue_size_ != 0) {
    auto trans_state_mask =
        finish_idx_.unsqueeze(1).expand({batch_size_, TRANS_HIDDEN_SIZE});
    for (int32_t layer = 0; layer < PRE_NUM_LAYERS; ++layer) {
      pre_hx_[layer].masked_fill_(trans_state_mask, 0);
      pre_cx_[layer].masked_fill_(trans_state_mask, 0.);
    }
    for (int32_t layer = 0; layer < POST_NUM_LAYERS; ++layer) {
      post_hx_[layer].masked_fill_(trans_state_mask, 0);
      post_cx_[layer].masked_fill_(trans_state_mask, 0.);
    }
    // update prediction tensors
    pre_g_.masked_fill_(finish_idx_, SOS);
    auto pred_state_mask =
        finish_idx_.unsqueeze(1).expand({batch_size_, PRED_HIDDEN_SIZE});
    for (int64_t layer = 0; layer < PRED_NUM_LAYERS; ++layer) {
      pre_hg_[layer].masked_fill_(pred_state_mask, 0.);
      pre_cg_[layer].masked_fill_(pred_state_mask, 0.);
    }
    // update res tensors
    res_idx_.masked_fill_(finish_idx_, -1);
  }
  // update f & f_lens
  F_lens_.masked_fill_(finish_idx_, 0);
  for (int32_t i = 0, j = 0; i < batch_size_ && j < dequeue_size; ++i) {
    if (finish_idx_[i].item().toBool()) {
      auto f_len = std::get<2>(dequeue_list[j]).item().toInt();
      F_lens_.index_put_({i}, f_len);
      remain_lens_.index_put_({i}, f_len);
      auto f = std::get<1>(dequeue_list[j]);
      F_.index_put_({Slice(0, f_len), i}, f.index({Slice(0, f_len)}).squeeze());
      samples[i] = std::get<0>(dequeue_list[j]);
      ++j;
    }
  }
  padded_size_ = batch_size_ - remain_size_ - dequeue_size;
  if (split_len_ > 0) {
    infer_lens_.zero_();
    finish_size_ = padded_size_;
    stop_size_ = std::min(batch_size_, padded_size_ + response_size_);
  } else {
    f_ = F_;
    f_lens_ = F_lens_;
    infer_lens_ = F_lens_;
    finish_idx_.fill_(1);
    finish_size_ = batch_size_;
  }
}

bool PipelineState::next() {
  bool status = (finish_size_ < stop_size_);
  if (status) {
    f_lens_ = torch::min(split_lens_, remain_lens_);
    TensorVector f_list;
    f_list.reserve(batch_size_);
    // gather f
    for (int32_t i = 0; i < batch_size_; ++i) {
      auto begin_idx_ = (remain_lens_[i].item().toInt() == 0)
                            ? 0
                            : (F_lens_[i] - remain_lens_[i]).item().toInt();
      f_list.emplace_back(
          F_.index({Slice(begin_idx_, begin_idx_ + split_len_), i, "..."}));
    }
    f_ = torch::stack(f_list, 1);
    remain_lens_ -= f_lens_;
    infer_lens_ += f_lens_;
    finish_idx_ = remain_lens_.le(0);
    finish_size_ = finish_idx_.count_nonzero().item().toInt();
  } else {
    remain_size_ = batch_size_ - finish_size_;
  }
  return status;
}

}  // namespace rnnt
