#include "utils.hpp"

#define NUM_THREADS 56
#define BUFFER_NUM_PER_INSTANCE 2

#define BACKBONE_SHAPE backbone_256_shape
#define SC_INIT_RN50_BACKBONE sc_init_rn50_backbone_bs256
#define RN50_BACKBONE rn50_backbone_bs256

#define TENSOR(NAME) \
    aligned_vector<float> NAME(product(BACKBONE_SHAPE::NAME()));

#define FILL_TENSOR(NAME) \
    uniform(-1.f, 1.f, product(BACKBONE_SHAPE::NAME()), \
            static_cast<float *>(NAME.data()));

// cat ../src/kernel_rn50/shape.hpp | grep "inline const" | awk -F"&" '{print
// $2}' | awk -F"(" '{print "TENSOR("$1 ");"}'
// plain inputs
std::vector<aligned_vector<int8_t>> backbone_input;
std::vector<aligned_vector<int8_t>> backbone_output;
TENSOR(res2a_bias_0);
TENSOR(res2a_bias_1);
TENSOR(res2a_bias_2);
TENSOR(res2a_bias_b);
TENSOR(res2a_weight_0);
TENSOR(res2a_weight_1);
TENSOR(res2a_weight_2);
TENSOR(res2a_weight_b);
TENSOR(res2b_bias_0);
TENSOR(res2b_bias_1);
TENSOR(res2b_bias_2);
TENSOR(res2b_weight_0);
TENSOR(res2b_weight_1);
TENSOR(res2b_weight_2);
TENSOR(res2c_bias_0);
TENSOR(res2c_bias_1);
TENSOR(res2c_bias_2);
TENSOR(res2c_weight_0);
TENSOR(res2c_weight_1);
TENSOR(res2c_weight_2);
TENSOR(res3a_bias_0);
TENSOR(res3a_bias_1);
TENSOR(res3a_bias_2);
TENSOR(res3a_bias_b);
TENSOR(res3a_weight_0);
TENSOR(res3a_weight_1);
TENSOR(res3a_weight_2);
TENSOR(res3a_weight_b);
TENSOR(res3b_bias_0);
TENSOR(res3b_bias_1);
TENSOR(res3b_bias_2);
TENSOR(res3b_weight_0);
TENSOR(res3b_weight_1);
TENSOR(res3b_weight_2);
TENSOR(res3c_bias_0);
TENSOR(res3c_bias_1);
TENSOR(res3c_bias_2);
TENSOR(res3c_weight_0);
TENSOR(res3c_weight_1);
TENSOR(res3c_weight_2);
TENSOR(res3d_bias_0);
TENSOR(res3d_bias_1);
TENSOR(res3d_bias_2);
TENSOR(res3d_weight_0);
TENSOR(res3d_weight_1);
TENSOR(res3d_weight_2);
TENSOR(res4a_bias_0);
TENSOR(res4a_bias_1);
TENSOR(res4a_bias_2);
TENSOR(res4a_bias_b);
TENSOR(res4a_weight_0);
TENSOR(res4a_weight_1);
TENSOR(res4a_weight_2);
TENSOR(res4a_weight_b);
TENSOR(res4b_bias_0);
TENSOR(res4b_bias_1);
TENSOR(res4b_bias_2);
TENSOR(res4b_weight_0);
TENSOR(res4b_weight_1);
TENSOR(res4b_weight_2);
TENSOR(res4c_bias_0);
TENSOR(res4c_bias_1);
TENSOR(res4c_bias_2);
TENSOR(res4c_weight_0);
TENSOR(res4c_weight_1);
TENSOR(res4c_weight_2);
TENSOR(res4d_bias_0);
TENSOR(res4d_bias_1);
TENSOR(res4d_bias_2);
TENSOR(res4d_weight_0);
TENSOR(res4d_weight_1);
TENSOR(res4d_weight_2);
TENSOR(res4e_bias_0);
TENSOR(res4e_bias_1);
TENSOR(res4e_bias_2);
TENSOR(res4e_weight_0);
TENSOR(res4e_weight_1);
TENSOR(res4e_weight_2);
TENSOR(res4f_bias_0);
TENSOR(res4f_bias_1);
TENSOR(res4f_bias_2);
TENSOR(res4f_weight_0);
TENSOR(res4f_weight_1);
TENSOR(res4f_weight_2);
TENSOR(res5a_bias_0);
TENSOR(res5a_bias_1);
TENSOR(res5a_bias_2);
TENSOR(res5a_bias_b);
TENSOR(res5a_weight_0);
TENSOR(res5a_weight_1);
TENSOR(res5a_weight_2);
TENSOR(res5a_weight_b);
TENSOR(res5b_bias_0);
TENSOR(res5b_bias_1);
TENSOR(res5b_bias_2);
TENSOR(res5b_weight_0);
TENSOR(res5b_weight_1);
TENSOR(res5b_weight_2);
TENSOR(res5c_bias_0);
TENSOR(res5c_bias_1);
TENSOR(res5c_bias_2);
TENSOR(res5c_weight_0);
TENSOR(res5c_weight_1);
TENSOR(res5c_weight_2);

static inline void run_with_single_instance(
        const int dst_idx = 0, const int src_idx = 0) {
    RN50_BACKBONE(backbone_output[dst_idx].data(),
            backbone_input[src_idx].data(), res2a_weight_b.data(),
            res2a_bias_b.data(), res2a_weight_0.data(), res2a_bias_0.data(),
            res2a_weight_1.data(), res2a_bias_1.data(), res2a_weight_2.data(),
            res2a_bias_2.data(), res2b_weight_0.data(), res2b_bias_0.data(),
            res2b_weight_1.data(), res2b_bias_1.data(), res2b_weight_2.data(),
            res2b_bias_2.data(), res2c_weight_0.data(), res2c_bias_0.data(),
            res2c_weight_1.data(), res2c_bias_1.data(), res2c_weight_2.data(),
            res2c_bias_2.data(), res3a_weight_b.data(), res3a_bias_b.data(),
            res3a_weight_0.data(), res3a_bias_0.data(), res3a_weight_1.data(),
            res3a_bias_1.data(), res3a_weight_2.data(), res3a_bias_2.data(),
            res3b_weight_0.data(), res3b_bias_0.data(), res3b_weight_1.data(),
            res3b_bias_1.data(), res3b_weight_2.data(), res3b_bias_2.data(),
            res3c_weight_0.data(), res3c_bias_0.data(), res3c_weight_1.data(),
            res3c_bias_1.data(), res3c_weight_2.data(), res3c_bias_2.data(),
            res3d_weight_0.data(), res3d_bias_0.data(), res3d_weight_1.data(),
            res3d_bias_1.data(), res3d_weight_2.data(), res3d_bias_2.data(),
            res4a_weight_b.data(), res4a_bias_b.data(), res4a_weight_0.data(),
            res4a_bias_0.data(), res4a_weight_1.data(), res4a_bias_1.data(),
            res4a_weight_2.data(), res4a_bias_2.data(), res4b_weight_0.data(),
            res4b_bias_0.data(), res4b_weight_1.data(), res4b_bias_1.data(),
            res4b_weight_2.data(), res4b_bias_2.data(), res4c_weight_0.data(),
            res4c_bias_0.data(), res4c_weight_1.data(), res4c_bias_1.data(),
            res4c_weight_2.data(), res4c_bias_2.data(), res4d_weight_0.data(),
            res4d_bias_0.data(), res4d_weight_1.data(), res4d_bias_1.data(),
            res4d_weight_2.data(), res4d_bias_2.data(), res4e_weight_0.data(),
            res4e_bias_0.data(), res4e_weight_1.data(), res4e_bias_1.data(),
            res4e_weight_2.data(), res4e_bias_2.data(), res4f_weight_0.data(),
            res4f_bias_0.data(), res4f_weight_1.data(), res4f_bias_1.data(),
            res4f_weight_2.data(), res4f_bias_2.data(), res5a_weight_b.data(),
            res5a_bias_b.data(), res5a_weight_0.data(), res5a_bias_0.data(),
            res5a_weight_1.data(), res5a_bias_1.data(), res5a_weight_2.data(),
            res5a_bias_2.data(), res5b_weight_0.data(), res5b_bias_0.data(),
            res5b_weight_1.data(), res5b_bias_1.data(), res5b_weight_2.data(),
            res5b_bias_2.data(), res5c_weight_0.data(), res5c_bias_0.data(),
            res5c_weight_1.data(), res5c_bias_1.data(), res5c_weight_2.data(),
            res5c_bias_2.data());
}

static float run_with_multiple_instances(const int times) {
    using namespace std::chrono;
    std::vector<float> elapsed(NUM_THREADS);
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
        auto start_time = steady_clock::now();
        for (int i = 0; i < times; i++) {
            run_with_single_instance(tid, tid);
            // tid, tid * BUFFER_NUM_PER_INSTANCE + i % 2);
        }
        elapsed[tid] = 1.f
                * duration_cast<microseconds>(steady_clock::now() - start_time)
                          .count()
                / (1000 * times);
    }
    float sum = 0.f;
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
        sum += elapsed[tid];
    }
    return sum / NUM_THREADS;
}

int main(int argc, char **argv) {
    int times = 20;
    if (argc > 1) { times = std::stoi(argv[1]); }

    // allocate in & out buffer for multi-instance scenario
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
        for (int i = 0; i < BUFFER_NUM_PER_INSTANCE; ++i) {
            backbone_input.emplace_back(aligned_vector<int8_t>(
                    product(BACKBONE_SHAPE::backbone_input()), 1.f));
        }
        backbone_output.emplace_back(aligned_vector<int8_t>(
                product(BACKBONE_SHAPE::backbone_output())));
    }

    // fill weights
    FILL_TENSOR(res2a_bias_0);
    FILL_TENSOR(res2a_bias_1);
    FILL_TENSOR(res2a_bias_2);
    FILL_TENSOR(res2a_bias_b);
    FILL_TENSOR(res2a_weight_0);
    FILL_TENSOR(res2a_weight_1);
    FILL_TENSOR(res2a_weight_2);
    FILL_TENSOR(res2a_weight_b);
    FILL_TENSOR(res2b_bias_0);
    FILL_TENSOR(res2b_bias_1);
    FILL_TENSOR(res2b_bias_2);
    FILL_TENSOR(res2b_weight_0);
    FILL_TENSOR(res2b_weight_1);
    FILL_TENSOR(res2b_weight_2);
    FILL_TENSOR(res2c_bias_0);
    FILL_TENSOR(res2c_bias_1);
    FILL_TENSOR(res2c_bias_2);
    FILL_TENSOR(res2c_weight_0);
    FILL_TENSOR(res2c_weight_1);
    FILL_TENSOR(res2c_weight_2);
    FILL_TENSOR(res3a_bias_0);
    FILL_TENSOR(res3a_bias_1);
    FILL_TENSOR(res3a_bias_2);
    FILL_TENSOR(res3a_bias_b);
    FILL_TENSOR(res3a_weight_0);
    FILL_TENSOR(res3a_weight_1);
    FILL_TENSOR(res3a_weight_2);
    FILL_TENSOR(res3a_weight_b);
    FILL_TENSOR(res3b_bias_0);
    FILL_TENSOR(res3b_bias_1);
    FILL_TENSOR(res3b_bias_2);
    FILL_TENSOR(res3b_weight_0);
    FILL_TENSOR(res3b_weight_1);
    FILL_TENSOR(res3b_weight_2);
    FILL_TENSOR(res3c_bias_0);
    FILL_TENSOR(res3c_bias_1);
    FILL_TENSOR(res3c_bias_2);
    FILL_TENSOR(res3c_weight_0);
    FILL_TENSOR(res3c_weight_1);
    FILL_TENSOR(res3c_weight_2);
    FILL_TENSOR(res3d_bias_0);
    FILL_TENSOR(res3d_bias_1);
    FILL_TENSOR(res3d_bias_2);
    FILL_TENSOR(res3d_weight_0);
    FILL_TENSOR(res3d_weight_1);
    FILL_TENSOR(res3d_weight_2);
    FILL_TENSOR(res4a_bias_0);
    FILL_TENSOR(res4a_bias_1);
    FILL_TENSOR(res4a_bias_2);
    FILL_TENSOR(res4a_bias_b);
    FILL_TENSOR(res4a_weight_0);
    FILL_TENSOR(res4a_weight_1);
    FILL_TENSOR(res4a_weight_2);
    FILL_TENSOR(res4a_weight_b);
    FILL_TENSOR(res4b_bias_0);
    FILL_TENSOR(res4b_bias_1);
    FILL_TENSOR(res4b_bias_2);
    FILL_TENSOR(res4b_weight_0);
    FILL_TENSOR(res4b_weight_1);
    FILL_TENSOR(res4b_weight_2);
    FILL_TENSOR(res4c_bias_0);
    FILL_TENSOR(res4c_bias_1);
    FILL_TENSOR(res4c_bias_2);
    FILL_TENSOR(res4c_weight_0);
    FILL_TENSOR(res4c_weight_1);
    FILL_TENSOR(res4c_weight_2);
    FILL_TENSOR(res4d_bias_0);
    FILL_TENSOR(res4d_bias_1);
    FILL_TENSOR(res4d_bias_2);
    FILL_TENSOR(res4d_weight_0);
    FILL_TENSOR(res4d_weight_1);
    FILL_TENSOR(res4d_weight_2);
    FILL_TENSOR(res4e_bias_0);
    FILL_TENSOR(res4e_bias_1);
    FILL_TENSOR(res4e_bias_2);
    FILL_TENSOR(res4e_weight_0);
    FILL_TENSOR(res4e_weight_1);
    FILL_TENSOR(res4e_weight_2);
    FILL_TENSOR(res4f_bias_0);
    FILL_TENSOR(res4f_bias_1);
    FILL_TENSOR(res4f_bias_2);
    FILL_TENSOR(res4f_weight_0);
    FILL_TENSOR(res4f_weight_1);
    FILL_TENSOR(res4f_weight_2);
    FILL_TENSOR(res5a_bias_0);
    FILL_TENSOR(res5a_bias_1);
    FILL_TENSOR(res5a_bias_2);
    FILL_TENSOR(res5a_bias_b);
    FILL_TENSOR(res5a_weight_0);
    FILL_TENSOR(res5a_weight_1);
    FILL_TENSOR(res5a_weight_2);
    FILL_TENSOR(res5a_weight_b);
    FILL_TENSOR(res5b_bias_0);
    FILL_TENSOR(res5b_bias_1);
    FILL_TENSOR(res5b_bias_2);
    FILL_TENSOR(res5b_weight_0);
    FILL_TENSOR(res5b_weight_1);
    FILL_TENSOR(res5b_weight_2);
    FILL_TENSOR(res5c_bias_0);
    FILL_TENSOR(res5c_bias_1);
    FILL_TENSOR(res5c_bias_2);
    FILL_TENSOR(res5c_weight_0);
    FILL_TENSOR(res5c_weight_1);
    FILL_TENSOR(res5c_weight_2);

    // init kernel
    SC_INIT_RN50_BACKBONE();

    // warmup
    for (int i = 0; i < 3; ++i) {
        run_with_single_instance(0, 0);
    }

    auto elapsed = run_with_multiple_instances(times);
    printf("Done rn50 backbone_bs256 %d iters with multiple instances in %.3f "
           "ms\n",
            times, elapsed);

    return 0;
}