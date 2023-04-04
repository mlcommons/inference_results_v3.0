#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

//#include "../cxxopts.hpp"
#include "amx_config.hpp"
#include "../amx_init.hpp"
#include "helper_test.h"
#include "helper.hpp"
#include "i_softmax_tpp.hpp"
#include "i_mha_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  int period = 5;
  int start = 0;
  auto t_ = reinterpret_cast<T (*)[cols]>(t);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (-- period == 0) {
        start += 1;
        period = 4;
      }
      t_[i][j] = start % 42;
    }
  }
}

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_SUPP 0x1021
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

#define ARCH_MAP_VDSO_X32 0x2001
#define ARCH_MAP_VDSO_32 0x2002
#define ARCH_MAP_VDSO_64 0x2003

inline bool amx_init() {
  unsigned long bitmask = 0;
  long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (0 != status)
    return false;
  if (bitmask & XFEATURE_MASK_XTILEDATA)
    return true;

  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (0 != status)
    return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                  // allowed
  status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
    return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
  return true;
}

// helpers to print a tile of array
void _i8(const void *arr, size_t rs, size_t rt, size_t cs, size_t ct,
    size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = rs; i < rt; ++i) {
    for (int j = cs; j < ct; ++j) {
      std::cout<<(int)t[i][j];
      if (j == ct -1)
        std::cout<<";";
      else
        std::cout<<",";
    }
    std::cout<<std::endl;
  }
}

void _i8_row(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int j = 0; j < col; ++j) {
    std::cout<<(int)t[row][j]<<",";
  }
  std::cout<<std::endl;
}

void _i8_col(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = 0; i < row; ++i) {
    std::cout<<(int)t[i][col]<<";";
    std::cout<<std::endl;
  }
}

void _i32(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      std::cout<<(int)t[i][j];
      if (j == col -1)
        std::cout<<";";
      else
        std::cout<<",";
    }
    std::cout<<std::endl;
  }
}

int main(int argc, char **argv) {
  //cxxopts::Options opts("mha_test", "MHA kernel test");
  //opts.add_options()
    //("l,seq_len", "Sequence length", cxxopts::value<size_t>()->default_value("384"))
    //("M,scale1", "First scale", cxxopts::value<float>()->default_value("0.0001"))
    //("s,oscale", "Second scale", cxxopts::value<float>()->default_value("8200"))
    //("f,eltscale", "Final scale", cxxopts::value<float>()->default_value("0.0001"))
    //("t,times", "Testing time", cxxopts::value<int>()->default_value("1"))
    //("b,batch", "Testing batch", cxxopts::value<int>()->default_value("64"))
  //;

  amx_init();

  auto parsed_opts = opts.parse(argc, argv);
  auto seq_len = parsed_opts["seq_len"].as<size_t>();
  auto M = parsed_opts["scale1"].as<float>();
  auto oscale = parsed_opts["oscale"].as<float>();
  auto M2 = parsed_opts["eltscale"].as<float>();
  auto times = parsed_opts["times"].as<int>();
  auto batch = parsed_opts["batch"].as<int>();

  void *attention;
  posix_memalign(&attention, 4096, batch * seq_len * 3072);
  fill_seq((int8_t *)attention, batch*seq_len, 3072);

  void *result;
  posix_memalign(&result, 4096, batch * seq_len * 1024);
  memset(result, 0, batch * seq_len * 1024);

  // Stepping in 64 and do all the heads
  auto b_att = reinterpret_cast<int8_t (*)[seq_len * 3072]>(attention);
  auto b_res = reinterpret_cast<int8_t (*)[seq_len * 1024]>(result);
  intel_mlperf::i_amx_mha_tpp mha(seq_len, 64);
  auto start = Time::now();

  intel_mlperf::Tilecfg __cfg;
  _tile_release();
  _tile_loadconfig(&__cfg.cfg);

  for (int t = 0; t < times; ++t)
  for (int b = 0; b < batch; ++b) {
    auto att = reinterpret_cast<int8_t (*)[64]>(b_att[b]);
    auto res = reinterpret_cast<int8_t (*)[64]>(b_res[b]);
#   pragma nounroll
    for (int i = 0; i < 16; ++ i) {
      mha.compute_head(res[i], att[i], 3072, M, oscale, M2);
    }
  }
  auto during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "Total time : "
            << (float)during / 1000 / 1000 << " ms " << std::endl;

  free(attention);
  free(result);

  return 0;
}
