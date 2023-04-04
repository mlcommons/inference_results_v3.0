// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <immintrin.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>
#include "nms.h"
//#include <csrc/autocast/autocast_mode.h>
//#include <csrc/jit/cpu/kernels/Softmax.h>

namespace torch_ipex {
/*
 When calculating the Intersection over Union:
  MaskRCNN: bias = 1
  SSD-Resnet34: bias = 0
*/
template <typename scalar_t, bool sorted>
at::Tensor nms_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold,
    float bias = 1.0);

template <typename scalar_t, bool sorted>
at::Tensor nms_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold,
    float bias) {
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  
  at::Tensor x_diff = x2_t.subtract(x1_t).add(bias);
  at::Tensor y_diff = y2_t.subtract(y1_t).add(bias);
  at::Tensor areas_t = x_diff.mul(y_diff);

  auto ndets = dets.size(0);
  // If scores and dets are already sorted in descending order, we don't need to
  // sort it again.
  auto order_t = sorted
      ? at::arange(0, ndets, scores.options().dtype(at::kLong))
      : std::get<1>(scores.sort(0, /* descending=*/true));

  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + bias);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + bias);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
    }
  }
  //return at::nonzero(suppressed_t == 0).squeeze(1);
  return at::nonzero(suppressed_t.eq(0)).squeeze(1);
}

#ifdef CPU_AVX512
// Optimized nms_cpu_kernel specialized for data type: float32 and sorted_score
template <>
at::Tensor nms_cpu_kernel</*scalar_t*/ float, /*sorted*/ true>(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold,
    float bias) {
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  AT_ASSERTM(dets.sizes().size() == 2, "dets should have 2 dimension");
  AT_ASSERTM(scores.sizes().size() == 1, "scores should have 1 dimension");

  at::Tensor dets_bbox_number_in_lastdim = dets.permute({1, 0}).contiguous();
  AT_ASSERTM(
      dets_bbox_number_in_lastdim.size(1) == scores.size(0),
      "dets should have number of bboxs as scores");
  AT_ASSERTM(
      dets_bbox_number_in_lastdim.size(0) == 4,
      "each bbox in dets should have 4 coordinates");

  if (dets_bbox_number_in_lastdim.numel() == 0) {
    return at::empty(
        {0},
        dets_bbox_number_in_lastdim.options().dtype(at::kLong).device(
            at::kCPU));
  }

  auto x1_t = dets_bbox_number_in_lastdim.select(0, 0).contiguous();
  auto y1_t = dets_bbox_number_in_lastdim.select(0, 1).contiguous();
  auto x2_t = dets_bbox_number_in_lastdim.select(0, 2).contiguous();
  auto y2_t = dets_bbox_number_in_lastdim.select(0, 3).contiguous();

  auto ndets = dets_bbox_number_in_lastdim.size(1);
  auto ndets_up_scale = (ndets / 16 + 1) * 16;
  auto ndets_down_scale = (ndets / 16) * 16;
  at::Tensor&& areas_t =
      at::zeros({ndets}, dets_bbox_number_in_lastdim.options()).contiguous();
  auto areas = areas_t.data_ptr<float>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  __m512 m512_zero = _mm512_setzero_ps();
  __m512 m512_bias = _mm512_set1_ps(bias);
  __m128i m128_zeroi = _mm_setzero_si128();

  // Step1: Calculate the area of all bbox's
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int i = 0; i < ndets_up_scale; i += 16) {
    __m512 m512_x1;
    __m512 m512_x2;
    __m512 m512_y1;
    __m512 m512_y2;
    __m512 m512_result;
    if (i < ndets_down_scale) {
      // vector
      m512_x1 = _mm512_loadu_ps(x1 + i);
      m512_x2 = _mm512_loadu_ps(x2 + i);
      m512_y1 = _mm512_loadu_ps(y1 + i);
      m512_y2 = _mm512_loadu_ps(y2 + i);
      if (bias == 0) {
        m512_result = _mm512_mul_ps(
            _mm512_sub_ps(m512_x2, m512_x1), _mm512_sub_ps(m512_y2, m512_y1));
      } else {
        m512_result = _mm512_mul_ps(
            _mm512_add_ps(_mm512_sub_ps(m512_x2, m512_x1), m512_bias),
            _mm512_add_ps(_mm512_sub_ps(m512_y2, m512_y1), m512_bias));
      }
      _mm512_storeu_ps(areas + i, m512_result);
    } else {
      // tail case
      unsigned short left_idx = ndets - ndets_down_scale;
      __mmask16 mask = (1 << left_idx) - 1; // 0x03ff;
      m512_x1 = _mm512_mask_loadu_ps(m512_zero, mask, x1 + i);
      m512_x2 = _mm512_mask_loadu_ps(m512_zero, mask, x2 + i);
      m512_y1 = _mm512_mask_loadu_ps(m512_zero, mask, y1 + i);
      m512_y2 = _mm512_mask_loadu_ps(m512_zero, mask, y2 + i);
      if (bias == 0) {
        m512_result = _mm512_mask_mul_ps(
            m512_zero,
            mask,
            _mm512_mask_sub_ps(m512_zero, mask, m512_x2, m512_x1),
            _mm512_mask_sub_ps(m512_zero, mask, m512_y2, m512_y1));
      } else {
        m512_result = _mm512_mask_mul_ps(
            m512_zero,
            mask,
            _mm512_mask_add_ps(
                m512_zero,
                mask,
                _mm512_mask_sub_ps(m512_zero, mask, m512_x2, m512_x1),
                m512_bias),
            _mm512_mask_add_ps(
                m512_zero,
                mask,
                _mm512_mask_sub_ps(m512_zero, mask, m512_y2, m512_y1),
                m512_bias));
      }
      _mm512_mask_storeu_ps(areas + i, mask, m512_result);
    }
  }
  // Step2: Go through the NMS flow
  at::Tensor suppressed_t = at::zeros(
      {ndets},
      dets_bbox_number_in_lastdim.options().dtype(at::kByte).device(at::kCPU));
  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  for (int64_t i = 0; i < ndets; i++) {
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    __m512 m512_ix1 = _mm512_set1_ps(ix1);
    __m512 m512_ix2 = _mm512_set1_ps(ix2);
    __m512 m512_iy1 = _mm512_set1_ps(iy1);
    __m512 m512_iy2 = _mm512_set1_ps(iy2);
    __m512 m512_iarea = _mm512_set1_ps(iarea);
    __m512 m512_threshold = _mm512_set1_ps(threshold);

    auto ndets_i_up_scale = ((ndets - i - 1) / 16 + 1) * 16;
    auto ndets_i_down_scale = ((ndets - i - 1) / 16) * 16;

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int64_t _j = 0; _j < ndets_i_up_scale; _j += 16) {
      if (_j < ndets_i_down_scale) {
        int64_t j = _j + i + 1;
        __m512 m512_x1 = _mm512_loadu_ps(x1 + j);
        __m512 m512_x2 = _mm512_loadu_ps(x2 + j);
        __m512 m512_y1 = _mm512_loadu_ps(y1 + j);
        __m512 m512_y2 = _mm512_loadu_ps(y2 + j);

        __m512 m512_xx1 = _mm512_max_ps(m512_ix1, m512_x1);
        __m512 m512_yy1 = _mm512_max_ps(m512_iy1, m512_y1);
        __m512 m512_xx2 = _mm512_min_ps(m512_ix2, m512_x2);
        __m512 m512_yy2 = _mm512_min_ps(m512_iy2, m512_y2);

        __m512 m512_w;
        __m512 m512_h;
        if (bias == 0) {
          m512_w = _mm512_max_ps(m512_zero, _mm512_sub_ps(m512_xx2, m512_xx1));
          m512_h = _mm512_max_ps(m512_zero, _mm512_sub_ps(m512_yy2, m512_yy1));
        } else {
          m512_w = _mm512_max_ps(
              m512_zero,
              _mm512_add_ps(_mm512_sub_ps(m512_xx2, m512_xx1), m512_bias));
          m512_h = _mm512_max_ps(
              m512_zero,
              _mm512_add_ps(_mm512_sub_ps(m512_yy2, m512_yy1), m512_bias));
        }

        __m512 m512_inter = _mm512_mul_ps(m512_w, m512_h);
        __m512 m512_areas = _mm512_loadu_ps(areas + j);
        __m512 m512_over = _mm512_div_ps(
            m512_inter,
            _mm512_sub_ps(_mm512_add_ps(m512_iarea, m512_areas), m512_inter));
        __mmask16 mask_sus =
            _mm512_cmp_ps_mask(m512_over, m512_threshold, _CMP_GE_OS);
        __m128i res_sus = _mm_mask_set1_epi8(m128_zeroi, mask_sus, 1);
        _mm_mask_storeu_epi8(suppressed + j, mask_sus, res_sus);

      } else {
        // Tail case
        int64_t j = _j + i + 1;
        int64_t idx_left = ndets - j;
        __mmask16 load_mask = (1 << idx_left) - 1;

        __m512 m512_x1 = _mm512_mask_loadu_ps(m512_zero, load_mask, x1 + j);
        __m512 m512_x2 = _mm512_mask_loadu_ps(m512_zero, load_mask, x2 + j);
        __m512 m512_y1 = _mm512_mask_loadu_ps(m512_zero, load_mask, y1 + j);
        __m512 m512_y2 = _mm512_mask_loadu_ps(m512_zero, load_mask, y2 + j);

        __m512 m512_xx1 =
            _mm512_mask_max_ps(m512_zero, load_mask, m512_ix1, m512_x1);
        __m512 m512_yy1 =
            _mm512_mask_max_ps(m512_zero, load_mask, m512_iy1, m512_y1);
        __m512 m512_xx2 =
            _mm512_mask_min_ps(m512_zero, load_mask, m512_ix2, m512_x2);
        __m512 m512_yy2 =
            _mm512_mask_min_ps(m512_zero, load_mask, m512_iy2, m512_y2);

        __m512 m512_w;
        __m512 m512_h;
        if (bias == 0) {
          m512_w = _mm512_mask_max_ps(
              m512_zero,
              load_mask,
              m512_zero,
              _mm512_mask_sub_ps(m512_zero, load_mask, m512_xx2, m512_xx1));
          m512_h = _mm512_mask_max_ps(
              m512_zero,
              load_mask,
              m512_zero,
              _mm512_mask_sub_ps(m512_zero, load_mask, m512_yy2, m512_yy1));
        } else {
          m512_w = _mm512_mask_max_ps(
              m512_zero,
              load_mask,
              m512_zero,
              _mm512_mask_add_ps(
                  m512_zero,
                  load_mask,
                  _mm512_mask_sub_ps(m512_zero, load_mask, m512_xx2, m512_xx1),
                  m512_bias));
          m512_h = _mm512_mask_max_ps(
              m512_zero,
              load_mask,
              m512_zero,
              _mm512_mask_add_ps(
                  m512_zero,
                  load_mask,
                  _mm512_mask_sub_ps(m512_zero, load_mask, m512_yy2, m512_yy1),
                  m512_bias));
        }
        __m512 m512_inter =
            _mm512_mask_mul_ps(m512_zero, load_mask, m512_w, m512_h);
        __m512 m512_areas =
            _mm512_mask_loadu_ps(m512_zero, load_mask, areas + j);
        __m512 m512_over = _mm512_mask_div_ps(
            m512_zero,
            load_mask,
            m512_inter,
            _mm512_mask_sub_ps(
                m512_zero,
                load_mask,
                _mm512_mask_add_ps(
                    m512_zero, load_mask, m512_iarea, m512_areas),
                m512_inter));
        __mmask16 mask_sus = _mm512_mask_cmp_ps_mask(
            load_mask, m512_over, m512_threshold, _CMP_GE_OS);
        __m128i res_sus = _mm_mask_set1_epi8(m128_zeroi, mask_sus, 1);
        _mm_mask_storeu_epi8(suppressed + j, mask_sus, res_sus);
      }
    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}
#endif

std::vector<at::Tensor> remove_empty(
    std::vector<at::Tensor>& candidate,
    int64_t start,
    int64_t end) {
  std::vector<at::Tensor> valid_candidate;
  for (int64_t i = start; i < end; i++) {
    if (candidate[i].defined()) {
      valid_candidate.push_back(candidate[i]);
    }
  }
  return valid_candidate;
}

template <typename scalar_t> std::vector<float> batch_score_nms_kernel(
    const at::Tensor& batch_dets,
    const at::Tensor& batch_scores,
    const at::Tensor& batch_labels,
    const int image_height,
    const int image_width,
    const float threshold,
    const int max_output = 300,
    std::vector<size_t> sample_index_list = {1}/* ids of images in the batch */) {
  
  // batch_dets: ( num_bbox, 4) For example: batch_dets: (120087, 4)
  // batch_scores: (num_bbox,) For example: batch_scores:( 0 < x < 128007,)
  auto nbatch = batch_scores.size(0); // number of batches (should be 1)
  at::Tensor unique_labels, _dummy;
  std::tie(unique_labels, _dummy) = at::_unique(batch_labels);//, false);

  int num_labels = unique_labels.size(0);
  float image_h = static_cast<float>(image_height);
  float image_w = static_cast<float>(image_height);

  //auto nbatch_x_nscore =
  //    nbatch * nscore; // (number of batches) * (number of labels)
  std::vector<at::Tensor> bboxes_out(num_labels);
  std::vector<at::Tensor> scores_out(num_labels);
  std::vector<at::Tensor> labels_out(num_labels);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif

  for (int index = 0; index < num_labels; index++) {
    // Parallel in the dimentaion of: labels of detected bboxes
    int i = unique_labels[index].item<int>(); // label

    at::Tensor label_loc = at::where(batch_labels.eq(i))[0]; // Get locations of this class
    
    at::Tensor bboxes = batch_dets.index_select(0, label_loc); // Get bboxes with this predicted class
    at::Tensor scores = batch_scores.index_select(0,label_loc);

    at::Tensor scores_sliced, scores_idx_sorted;
    std::tie(scores_sliced, scores_idx_sorted) = at::topk(
        scores, (max_output > scores.size(0)) ? scores.size(0) : max_output, 0);

    at::Tensor bboxes_sliced = at::index_select(bboxes, /*dim*/ 0, scores_idx_sorted);

    at::Tensor keep = nms_cpu_kernel<scalar_t, /*sorted*/ true>(
        bboxes_sliced, scores_sliced, threshold, /*bias*/ 0);


    bboxes_out[index] = at::index_select(bboxes_sliced, /*dim*/ 0, keep);
    scores_out[index] = at::index_select(scores_sliced, /*dim*/ 0, keep);

    // TODO optimize the fill_
    labels_out[index] = at::empty({keep.sizes()}).fill_(i);
  }

  at::Tensor bboxes_out_, labels_out_, scores_out_;

  // TODO: Change to max_output
  if (bboxes_out.size() > 0 /* max_output */){
    bboxes_out_ = at::cat(bboxes_out, 0);
    labels_out_ = at::cat(labels_out, 0);
    scores_out_ = at::cat(scores_out, 0);

    /* If we have more predictions than max_output */
    // TODO: Change to max_output
    if (scores_out_.size(0) > 0){//max_output){
      at::Tensor sort_idxs, sort_result;
      std::tie(sort_result, sort_idxs) = scores_out_.sort(0, true); // sort in descending
      sort_idxs = sort_idxs.slice(0, 0, std::min(sort_idxs.size(0), static_cast<int64_t>(max_output)));

      bboxes_out_ = bboxes_out_.index_select(0, sort_idxs); // Select the top-max-output bboxes
      labels_out_ = labels_out_.index_select(0, sort_idxs);
      scores_out_ = sort_result.slice(0,0,max_output).sigmoid(); //Finally take sigmoid of scores (also sort_result is already sorted)
    }
  }

  /* Add post-processor for loadgen */
  
  int64_t proposal_count = bboxes_out_.size(0);
  //std::vector<at::Tensor> processed_results;// = at::empty({proposal_count, 7});
  std::vector<float> processed_results;
  processed_results.resize(proposal_count * 7);// Each detected object has 7 descriptors to loadgen

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int proposal = 0; proposal < proposal_count; proposal++){
    at::Tensor detection = at::ones(7);

    processed_results[proposal*7] = static_cast<float>(sample_index_list[0]);
    processed_results[proposal*7 + 1] = bboxes_out_[proposal][1].item<float>() / image_h;
    processed_results[proposal*7 + 2] = bboxes_out_[proposal][0].item<float>() / image_w;
    processed_results[proposal*7 + 3] = bboxes_out_[proposal][3].item<float>() / image_h;
    processed_results[proposal*7 + 4] = bboxes_out_[proposal][2].item<float>() / image_w;
    processed_results[proposal*7 + 5] = scores_out_[proposal].item<float>();
    processed_results[proposal*7 + 6] = labels_out_[proposal].item<float>();

    //processed_results[proposal] = detection.clone();
  }
  //std::cout << " ** Processed results size: " << processed_results.size() << std::endl;  
  return processed_results; 
  //return at::stack(processed_results);
}


/*
std::vector<float> batch_score_nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float threshold,
    const int max_output,
    std::vector<size_t> sample_index_list) {
  std::vector<float> result;

  /
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "batch_score_nms", [&] {
    result =
        batch_score_nms_kernel<scalar_t>(dets, scores, labels, threshold, max_output, sample_index_list);
  });
  /
  result = batch_score_nms_kernel<float>(dets, scores, labels, threshold, max_output, sample_index_list);
  return result;
}
*/

std::vector<float> batch_score_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const int image_height, // preprocessed_image_height (retinanet->800)
    const int image_width,
    const double threshold,
    const int64_t max_output,
    std::vector<size_t> sample_index_list) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "IpexExternal::batch_score_nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  return batch_score_nms_kernel<float>(dets, scores, labels, image_height, image_width, threshold, max_output, sample_index_list);
}


} // namespace torch_ipex

