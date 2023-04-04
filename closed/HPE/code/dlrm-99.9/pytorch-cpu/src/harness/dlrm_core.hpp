#pragma once
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>
#include <glog/logging.h>
#include "kernels.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "util.hpp"

class DLRMParam {
public:
    DLRMParam(float *param, size_t len);
    ~DLRMParam();
    void quantize(std::vector<size_t> &dims, bool per_tensor);
    void s32quantize(std::vector<size_t> &dims, std::vector<float> scales);
    // weight and bias scale
    std::vector<float> GetScale();
    int8_t *qdata();
    int *s32data();
private:
    const float *param_;
    int8_t *qparam_;
    int32_t *s32param_;
    std::vector<float> scales_;
    size_t len_;
};

using QParam_t = std::map<std::string, std::shared_ptr<DLRMParam>>;

template <bool constM, int64_t MB>
dnnl::matmul::primitive_desc matmul_pd_create(
    int64_t M, int64_t K, int64_t N, const dnnl::engine &eng, bool relu = true) {
    const int64_t Mfinal = constM ? MB : (M % MB);
    dnnl::memory::desc a_md({Mfinal, K}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
    dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
    auto c_dtype = dnnl::memory::data_type::s8;
    if (!relu) {
        c_dtype = dnnl::memory::data_type::f32;
    }
    dnnl::memory::desc c_md({Mfinal, N}, c_dtype, dnnl::memory::format_tag::ab);
    dnnl::memory::desc bias_md({1, N}, dnnl::memory::data_type::s32, {N, 1});

    dnnl::primitive_attr attr;
    attr.set_output_scales(2, {DNNL_RUNTIME_F32_VAL});
    attr.set_zero_points(DNNL_ARG_SRC, 0, {0});
    attr.set_zero_points(DNNL_ARG_DST, 0, {0});
    if (relu) {
        dnnl::post_ops po;
        po.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
        attr.set_post_ops(po);
    }
    else {
        dnnl::post_ops po;
        po.append_eltwise(1.0f, dnnl::algorithm::eltwise_logistic, 0.0f, 0.0f);
        attr.set_post_ops(po);
    }

    dnnl::matmul::desc matmul_d(a_md, b_md, bias_md, c_md);
    dnnl::matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    return matmul_pd;
}

extern std::unordered_map<int, dnnl::memory> B_m_pack;
extern std::unordered_map<int, dnnl::memory> bias_m;

template <int64_t MB, int LAYOFS>
struct fusemlp {
    dnnl::engine eng;
    dnnl::stream s;
    std::vector<dnnl::matmul> matmul_p_MB;
    std::vector<dnnl::matmul> matmul_p_RS;
    std::vector<dnnl::memory> scales_m;

    int8_t *buff0 = nullptr;
    int8_t *buff1 = nullptr;
    int64_t last_batch = -1;
    int OFFSET = 0;
    fusemlp(int64_t n0, int64_t n1, int local_offset) {
        this->eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        this->s = dnnl::stream(this->eng);
        if (this->buff0 == nullptr)
            CHECK_EQ(mallocHost((void **)&(this->buff0), MB * n0 * sizeof(int8_t)), cpuSucc);
        // this->buff0 = new_data<int8_t>(MB * n0 * sizeof(int8_t));
        if (this->buff1 == nullptr)
            CHECK_EQ(mallocHost((void **)&(this->buff1), MB * n1 * sizeof(int8_t)), cpuSucc);
        // this->buff1 = new_data<int8_t>(MB * n1 * sizeof(int8_t));
        OFFSET = LAYOFS + local_offset * 16;
    }

    ~fusemlp() {
        free(buff0);
        free(buff1);
        buff0 = nullptr;
        buff1 = nullptr;
    }

    template<int NLAY>
    void setup(const int64_t *M,
               const int64_t *N,
               const int64_t *K,
               std::vector<std::vector<float>> &scales,
               int8_t **weight_int8,
               int **bias_int32,
               int sigmoid_layer = -1) {
        // static_assert(NLAY > 0 && "NLAY must be a positive integer");
        for (int i = 0; i < NLAY; ++i) {
            bool relu = true;
            if (i == sigmoid_layer) {
                relu = false;
            }
            dnnl::matmul::primitive_desc matmul_pd =
                matmul_pd_create<true, MB>(M[i], K[i], N[i], eng, relu);
            this->matmul_p_MB.push_back(dnnl::matmul(matmul_pd));
            if (bias_m.find(OFFSET + i) == bias_m.end())
                bias_m[OFFSET + i] =
                    dnnl::memory(matmul_pd.bias_desc(), eng, (void *)bias_int32[i]);
            if (B_m_pack.find(OFFSET + i) == B_m_pack.end())
                B_m_pack[OFFSET + i] =
                    dnnl::memory(matmul_pd.weights_desc(), eng);
            this->scales_m.push_back(
                dnnl::memory(
                    {{N[i]}, dnnl::memory::data_type::f32,
                     {1}},
                    eng, (void *)scales[i].data()));
            this->prepack(K[i], N[i], weight_int8[i], B_m_pack[i + OFFSET]);
        }
        this->restMpd<NLAY>(M, N, K, sigmoid_layer);
    }

    template<int NLAY>
    void restMpd(const int64_t *M,
                 const int64_t *N,
                 const int64_t *K,
                 int sigmoid_layer=-1) {
        // static_assert(NLAY > 0 && "NLAY must be a positive integer");
        if (this->last_batch != M[0]) {
            this->matmul_p_RS.clear();
            if (M[0] % MB > 0) {
                for (int i = 0; i < NLAY; ++i) {
                    bool relu = true;
                    if (i == sigmoid_layer) {
                        relu = false;
                    }
                    dnnl::matmul::primitive_desc matmul_pd =
                        matmul_pd_create<false, MB>(M[i], K[i], N[i],
                                                    this->eng, relu);
                    this->matmul_p_RS.push_back(dnnl::matmul(matmul_pd));
                }
            }
            this->last_batch = M[0];
        }
    }

    void prepack(int64_t K, int64_t N, const int8_t *src, dnnl::memory &dst) {
        dnnl::memory orig_mem(
            {{K, N},
             dnnl::memory::data_type::s8,
             dnnl::memory::format_tag::ba},
            this->eng, (void *)src);
        dnnl::reorder(orig_mem, dst).execute(this->s, orig_mem, dst);
        this->s.wait();
    }

    template<int NLAY, typename T>
    void fusemlp_dnnpack_ptr(const int8_t *input,
                             T *res,
                             const int64_t *M, const int64_t *N,
                             const int64_t *K,
                             std::vector<std::vector<float>> &scales,
                             int sigmoid_layer=-1) {
        // static_assert(NLAY > 0 && "NLAY must be a positive integer");
        int64_t restM = M[0] % MB;
        int64_t MM = M[0] / MB;
        dnnl::memory::desc a_mds[NLAY];
        dnnl::memory::desc c_mds[NLAY];
        this->restMpd<NLAY>(M, N, K, sigmoid_layer);
        for (int i = 0; i < NLAY; ++i) {
            a_mds[i] = dnnl::memory::desc({MB, K[i]}, dnnl::memory::data_type::s8,
                                          dnnl::memory::format_tag::ab);
            auto c_dtype = dnnl::memory::data_type::s8;
            if (i == sigmoid_layer) {
                c_dtype = dnnl::memory::data_type::f32;
            }
            c_mds[i] = dnnl::memory::desc({MB, N[i]}, c_dtype,
                                          dnnl::memory::format_tag::ab);
        }

        for (int mm = 0; mm < MM; ++mm) {
            for (int i = 0; i < NLAY - 1; ++i) {
                const int8_t *src = (i == 0) ? &input[mm * MB * K[0]] : (i % 2 == 0 ? buff1 : buff0);
                int8_t *dst = (i % 2 == 0 ? buff0 : buff1);
                dnnl::memory A_m(a_mds[i], this->eng, (void *)src);
                dnnl::memory C_m(c_mds[i], this->eng, dst);
                dnnl::memory scales_m({{N[i]}, dnnl::memory::data_type::f32, {1}},
                                      this->eng, (void *)scales[i].data());
                this->matmul_p_MB[i].execute(
                    this->s,
                    {{DNNL_ARG_SRC, A_m},
                     {DNNL_ARG_WEIGHTS, B_m_pack[i + OFFSET]},
                     {DNNL_ARG_BIAS, bias_m[i + OFFSET]},
                     {DNNL_ARG_DST, C_m},
                     {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m}});
            }
            int i = NLAY - 1;
            const int8_t *src = (i == 0) ? &input[mm * MB * K[0]] : (i % 2 == 0 ? buff1 : buff0);
            T *dst = &res[mm * MB * N[NLAY-1]];
            dnnl::memory A_m(a_mds[i], this->eng, (void *)src);
            dnnl::memory C_m(c_mds[i], this->eng, (void *)dst);
            dnnl::memory scales_m({{N[i]}, dnnl::memory::data_type::f32, {1}},
                                  this->eng, (void *)scales[i].data());
            this->matmul_p_MB[i].execute(
                this->s,
                {{DNNL_ARG_SRC, A_m},
                 {DNNL_ARG_WEIGHTS, B_m_pack[i + OFFSET]},
                 {DNNL_ARG_BIAS, bias_m[i + OFFSET]},
                 {DNNL_ARG_DST, C_m},
                 {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m}});

        }

        if (restM > 0) {
            for (int i = 0; i < NLAY - 1; ++i) {
                const int8_t *src = (i == 0) ? &input[MM * MB * K[0]] : (i % 2 == 0 ? buff1 : buff0);
                int8_t *dst = (i % 2 == 0 ? buff0 : buff1);
                // int8_t *dst = (i == NLAY-1) ? &res[MM * MB * N[NLAY-1]] : (i % 2 == 0 ? buff0 : buff1);
                dnnl::memory::desc a_md({restM, K[i]}, dnnl::memory::data_type::s8,
                                        dnnl::memory::format_tag::ab);
                auto c_dtype = dnnl::memory::data_type::s8;
                if (i == sigmoid_layer) {
                    c_dtype = dnnl::memory::data_type::f32;
                }
                dnnl::memory::desc c_md({restM, N[i]}, c_dtype,
                                        dnnl::memory::format_tag::ab);
                dnnl::memory A_m(a_md, this->eng, (void *)src);
                dnnl::memory C_m(c_md, this->eng, (void *)dst);
                dnnl::memory scales_m({{N[i]}, dnnl::memory::data_type::f32, {1}},
                                      this->eng, (void *)scales[i].data());
                this->matmul_p_RS[i].execute(
                    this->s,
                    {{DNNL_ARG_SRC, A_m},
                     {DNNL_ARG_WEIGHTS, B_m_pack[i + OFFSET]},
                     {DNNL_ARG_BIAS, bias_m[i + OFFSET]},
                     {DNNL_ARG_DST, C_m},
                     {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m}});
            }

            int i = NLAY - 1;
            const int8_t *src = (i == 0) ? &input[MM * MB * K[0]] : (i % 2 == 0 ? buff1 : buff0);
            T *dst = &res[MM * MB * N[NLAY-1]];
            dnnl::memory::desc a_md({restM, K[i]}, dnnl::memory::data_type::s8,
                                    dnnl::memory::format_tag::ab);
            auto c_dtype = dnnl::memory::data_type::s8;
            if (i == sigmoid_layer) {
                c_dtype = dnnl::memory::data_type::f32;
            }
            dnnl::memory::desc c_md({restM, N[i]}, c_dtype,
                                    dnnl::memory::format_tag::ab);
            dnnl::memory A_m(a_md, this->eng, (void *)src);
            dnnl::memory C_m(c_md, this->eng, (void *)dst);
            dnnl::memory scales_m({{N[i]}, dnnl::memory::data_type::f32, {1}},
                                  this->eng, (void *)scales[i].data());
            this->matmul_p_RS[i].execute(
                this->s,
                {{DNNL_ARG_SRC, A_m},
                 {DNNL_ARG_WEIGHTS, B_m_pack[i + OFFSET]},
                 {DNNL_ARG_BIAS, bias_m[i + OFFSET]},
                 {DNNL_ARG_DST, C_m},
                 {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m}});
        }
    }
};


using fusebotmlp = fusemlp<1024, 0>;
using fusetopmlp = fusemlp<1024, 3>;

class Embint {
public:
    Embint(QParam_t &params,
           const float xin_scale,
           const float out_scale)
        : c_scale_(out_scale / xin_scale),
          tc_{0} {
        std::vector<float> in_scales(_S1);
        std::vector<std::string> wgt_name(_S);
        for (int i = 0; i < 26; ++i) {
            wgt_name[i] = std::string("emb_l.") + std::to_string(i) + std::string(".weight");
        };

        for (int i = 0; i < _S; ++i) {
            auto p = params[wgt_name[i]];
            weights_[i] = p->qdata();
            in_scales[i + 1] = p->GetScale()[0];
        }
        in_scales[0] = xin_scale;

        const float r_scale = out_scale;

        size_t offset = 0;
        for (int i = 1; i < _S1; ++i) {
            for (int j = 0; j < i; ++j) {
                scales_[offset] = r_scale / (in_scales[i] * in_scales[j]);
                offset++;
            }
        }
        for (int i = 351; i < 384; ++i)
            scales_[i] = 0.0f;

        for (int i = 0; i < 8; ++i) {
            tc_.colb[i] = 64;
            tc_.rows[i] = 16;
        }
        tc_.palette_id = 1;

        do_dense_scale_ = (std::abs(c_scale_ - 1.0) > 0.0005);
    }

    void forward(const size_t Batch,
                 const int8_t *botout,
                 const int32_t *index,
                 int8_t *intout) {
        if (do_dense_scale_)
            fuse_emb_int_kernel<_K, ROW, true>(Batch,
                                               &tc_,
                                               botout,
                                               index,
                                               weights_,
                                               intout,
                                               c_scale_,
                                               scales_);
        else
            fuse_emb_int_kernel<_K, ROW, false>(Batch,
                                                &tc_,
                                                botout,
                                                index,
                                                weights_,
                                                intout,
                                                -1.0f,
                                                scales_);
    }
    ~Embint() {}
    static constexpr uint8_t _S = 26;
    static constexpr uint8_t _S1 = 27;
    static constexpr uint8_t _M = 32;
    static constexpr uint8_t TILE_M = 16;
    static constexpr uint8_t TILE_N = TILE_M;
    static constexpr uint8_t TILE_K = 64;
    static constexpr uint8_t TILE_BROWS = 16;
    static constexpr uint8_t _K = 128;
    static constexpr uint8_t LOG2_K = 7;
    static constexpr int ROW = 512;
private:
    const float c_scale_;
    float scales_[384] __attribute__((aligned(64)));
    const int8_t * weights_[_S];
    tileconfig tc_{0};
    bool do_dense_scale_;
};

class DLRMCore {
public:
    DLRMCore(QParam_t &params, std::vector<float> &scales, int prod_idx);
    ~DLRMCore();
    float *infer(size_t batch, int8_t *input, int32_t *lsi);
    float getScale(int op_id, bool inp);

    // get a output buffer from pool
    template<typename T>
    T *getBuffer(size_t batch, size_t inner_dim,
                 std::map<size_t, T *> &buffer_pool) {
        if (buffer_pool.find(batch) == buffer_pool.end()) {
            // create a buffer
            T *ptr = nullptr;
            CHECK_EQ(mallocHost((void **)&ptr, batch * inner_dim * sizeof(T)),
                     cpuSucc);
            // save the buffer
            buffer_pool[batch] = ptr;
        }
        return buffer_pool[batch];
    }
private:
    // botmlp fusemlp
    std::shared_ptr<fusebotmlp> botmlp_;
    // botmlp buffer
    std::map<size_t, int8_t *> botmlpbuf_;
    // botmlp scales
    std::vector<std::vector<float>> botscales_;

    // for topmlp
    std::shared_ptr<fusetopmlp> topmlp_;
    // topmlp buffer
    std::map<size_t, float *> topmlpbuf_;
    // topmlp scales
    std::vector<std::vector<float>> topscales_;

    // interaction buffer
    std::shared_ptr<Embint> fuseembint_;
    std::map<size_t, int8_t *> intertbuf_;
    // interaction scales
    std::vector<float> intscales_;

    QParam_t params_;
    std::vector<float> scales_;
};
