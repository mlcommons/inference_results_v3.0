#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <thread>
#include "cnpy.h"
#include "dlrm_core.hpp"
#include "util.hpp"

inline float
absmax(const float *p, const size_t len) {
    float res = 0.0f;
// #pragma omp parallel for reduction(max:res)
    for (size_t i = 0; i < len; ++i) {
        res = std::max(std::abs(p[i]), res);
    }
    return res;
}

inline void
fp32tos8(const float *src, int8_t *dst, const float scale, const size_t len) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
        float s = src[i] * scale;
        if (s >= 127.5)
            s = 127.0f;
        // CHECK_GT(s, -128.0f);
        // CHECK_LT(s, 127.5);
        dst[i] = (int8_t)round(s);
    }
}

DLRMParam::DLRMParam(float *param, size_t len)
    :param_(param), qparam_(nullptr), s32param_(nullptr), len_(len) {
}


DLRMParam::~DLRMParam() {
    if (qparam_ != nullptr)
        CHECK_EQ(freeHost((void **)&qparam_), cpuSucc);
    if (s32param_ != nullptr)
        CHECK_EQ(freeHost((void **)&s32param_), cpuSucc);
}

void DLRMParam::quantize(std::vector<size_t> &dims, bool per_tensor) {
    CHECK_NE(param_, nullptr);
    if (qparam_ != nullptr)
        return;
    CHECK_EQ(s32param_, nullptr);
    CHECK_GT(len_, 0);
    // allocate buffer
    CHECK_EQ(mallocHost((void **)&qparam_, len_ * sizeof(int8_t)), cpuSucc);
    CHECK_EQ(dims.size(), 2);
    if (per_tensor) {
        CHECK_EQ(dims[0] * dims[1], len_);
        float scale = 127.5 / absmax(param_, len_);
        fp32tos8(param_, qparam_, scale, len_);
        scales_.push_back(scale);
    } else {
        CHECK_EQ(dims[0] * dims[1], len_);
        // LOG(INFO) << "quantize " << dims[0] << " row " << dims[1] << " elem per row";
        for (int i = 0; i < dims[0]; ++i) {
            float scale = 127.5 / absmax(param_ + i * dims[1], dims[1]);
            fp32tos8(param_ + i * dims[1], qparam_ + i * dims[1], scale, dims[1]);
            scales_.push_back(scale);
        }
    }
}

void DLRMParam::s32quantize(std::vector<size_t> &dims, std::vector<float> scales) {
    CHECK_NE(param_, nullptr);
    if (s32param_ != nullptr)
        return;
    CHECK_EQ(qparam_, nullptr);
    CHECK_EQ(len_, scales.size());
    // allocate buffer
    CHECK_EQ(mallocHost((void **)&s32param_, len_ * sizeof(int32_t)), cpuSucc);
    CHECK_EQ(dims.size(), 1);
    CHECK_EQ(scales.size(), dims[0]);
    for (int i = 0; i < len_; ++i) {
        float s = scales[i];
        float p = param_[i];
        s32param_[i] = (int32_t)round(s * p);
    }
}

std::vector<float>
DLRMParam::GetScale() {
    return scales_;
}

int8_t *
DLRMParam::qdata() {
    CHECK_NE(qparam_, nullptr);
    return qparam_;
}

int32_t *
DLRMParam::s32data() {
    CHECK_NE(s32param_, nullptr);
    return s32param_;
}

std::unordered_map<int, dnnl::memory> B_m_pack;
std::unordered_map<int, dnnl::memory> bias_m;

DLRMCore::DLRMCore(QParam_t &params, std::vector<float> &scales, int prod_idx)
    :params_(params), scales_(scales) {
    // setup botmlp
    {
        int64_t M[3] = {1024, 1024, 1024};
        int64_t N[3] = {512, 256, 128};
        int64_t K[3] = {32, 512, 256};
        std::vector<std::string> botwgt_names = {
            "bot_l.0.weight",
            "bot_l.2.weight",
            "bot_l.4.weight"};
        std::vector<std::string> botbias_names = {
            "bot_l.0.bias",
            "bot_l.2.bias",
            "bot_l.4.bias"
        };
        // setup onednn wrapper
        botmlp_ = std::shared_ptr<fusebotmlp>(new fusebotmlp(N[0], N[1], prod_idx));

        int8_t *w_ptrs[3];
        int32_t *b_ptrs[3];
        // setup scales
        for (int i = 0; i < 3; ++i) {
            // calculate weight scales for onednn
            auto scalew = params[botwgt_names[i]]->GetScale();
            std::vector<float> scale_per_l;
            float input_scale = getScale(i, true);
            float output_scale = getScale(i, false);
            for (auto s : scalew) {
                scale_per_l.push_back(output_scale / (input_scale * s));
            }
            botscales_.push_back(scale_per_l);
            // calculate bias scales
            std::vector<float> biasscales;
            for (auto v : scalew)
                biasscales.push_back(v * getScale(i, true));
            std::vector<size_t> biasdim;
            biasdim.push_back(N[i]);
            // quantize bias
            params[botbias_names[i]]->s32quantize(biasdim, biasscales);
            w_ptrs[i] = params[botwgt_names[i]]->qdata();
            b_ptrs[i] = params[botbias_names[i]]->s32data();
        }
        botmlp_->setup<3>(M, N, K, botscales_, w_ptrs, b_ptrs, -1);
    }
    {
        // setup interaction
        fuseembint_ = std::make_shared<Embint>(Embint(params, getScale(3, true), getScale(3, false)));
    }
    // setup topmlp
    {
        int64_t M[5] = {1024, 1024, 1024, 1024, 1024};
        int64_t N[5] = {1024, 1024, 512, 256, 1};
        int64_t K[5] = {512, 1024, 1024, 512, 256};
        std::vector<std::string> topwgt_names = {
            "top_l.0.weight",
            "top_l.2.weight",
            "top_l.4.weight",
            "top_l.6.weight",
            "top_l.8.weight"
        };
        std::vector<std::string> topbias_names = {
            "top_l.0.bias",
            "top_l.2.bias",
            "top_l.4.bias",
            "top_l.6.bias",
            "top_l.8.bias"
        };
        // setup onednn wrapper
        topmlp_ = std::shared_ptr<fusetopmlp>(new fusetopmlp(N[0], N[1], prod_idx));
        constexpr int sigmoid_layer = 4;
        // setup scales
        int8_t *w_ptrs[5];
        int32_t *b_ptrs[5];
        const int layer = 4;
        for (int i = 0; i < 5; ++i) {
            // calculate weight scales
            auto scalew = params[topwgt_names[i]]->GetScale();
            std::vector<float> scale_per_l;
            float input_scale = getScale(4 + i, true);
            float output_scale = getScale(4 + i, false);
            if (i == sigmoid_layer) {
                for (auto s : scalew) {
                    scale_per_l.push_back(1.0f / (input_scale * s));
                }
            } else {
                for (auto s : scalew) {
                    scale_per_l.push_back(output_scale / (input_scale * s));
                }
            }
            topscales_.push_back(scale_per_l);
            // calculate bias scales
            std::vector<float> biasscales;
            for (auto v : scalew)
                biasscales.push_back(v * getScale(4 + i, true));
            std::vector<size_t> biasdim;
            biasdim.push_back(N[i]);
            // quantize bias
            params[topbias_names[i]]->s32quantize(biasdim, biasscales);
            w_ptrs[i] = params[topwgt_names[i]]->qdata();
            b_ptrs[i] = params[topbias_names[i]]->s32data();
        }
        topmlp_->setup<5>(M, N, K, topscales_, w_ptrs, b_ptrs, sigmoid_layer);
    }
}

DLRMCore::~DLRMCore() {
    // clean buffer
    for (auto &b : botmlpbuf_) {
        CHECK_EQ(freeHost((void **)&b.second), cpuSucc);
    }
    for (auto &t : topmlpbuf_) {
        CHECK_EQ(freeHost((void **)&t.second), cpuSucc);
    }
    for (auto &i : intertbuf_) {
        CHECK_EQ(freeHost((void **)&i.second), cpuSucc);
    }
}

float *DLRMCore::infer(size_t batch, int8_t *dsx, int32_t *lsi) {
    int8_t *bot_out = getBuffer(batch, 128, botmlpbuf_);
    int8_t *int_out = getBuffer(batch, 512, intertbuf_);
    float *top_out = getBuffer(batch, 1, topmlpbuf_);
    int64_t botM[3] = {static_cast<int64_t>(batch),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(batch)};
    int64_t botN[3] = {512, 256, 128};
    int64_t botK[3] = {32, 512, 256};

    int64_t topM[5] = {static_cast<int64_t>(batch),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(batch)};
    int64_t topN[5] = {1024, 1024, 512, 256, 1};
    int64_t topK[5] = {512, 1024, 1024, 512, 256};
    constexpr int sigmoid_layer = 4;
    botmlp_->restMpd<3>(botM, botN, botK);
    topmlp_->restMpd<5>(topM, topN, topK, sigmoid_layer);

    // botmlp forward
    botmlp_->fusemlp_dnnpack_ptr<3, int8_t>(dsx, bot_out,
                                            botM, botN, botK,
                                            botscales_);

    // interaction
    fuseembint_->forward(batch, bot_out, lsi, int_out);

    // topmlp forward
    topmlp_->fusemlp_dnnpack_ptr<5, float>(int_out, top_out,
                                           topM, topN, topK,
                                           topscales_, sigmoid_layer);
    return top_out;
}

float DLRMCore::getScale(int op_id, bool inp) {
    return inp ? scales_[op_id * 2 + 0] : scales_[op_id * 2 + 1];
}
