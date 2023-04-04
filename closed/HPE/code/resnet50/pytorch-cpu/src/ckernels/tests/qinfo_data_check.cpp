#include <chrono>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <vector>

using namespace std;

// 222400 / 8
extern uint64_t rn50_backbone_bs9_data[27800];
extern uint64_t qinfos_data[26760];

int main() {
  size_t full_data_size = sizeof(rn50_backbone_bs9_data);
  size_t qinfos_data_size = sizeof(qinfos_data);

  size_t full_data_elems = full_data_size / sizeof(uint64_t);
  size_t qinfos_data_elems = qinfos_data_size / sizeof(uint64_t);

  std::cout << "full_data_size=" << full_data_size
            << ", qinfos_data_size=" << qinfos_data_size << std::endl;

  size_t offset = full_data_elems - qinfos_data_elems;
#define OFFSET 26760
  //#define OFFSET 13930
  //#define OFFSET 13420 //includes last two dequantize
  //#define OFFSET 13248
  // weight 13248
  size_t off1 = full_data_elems - OFFSET;
  size_t off2 = qinfos_data_elems - OFFSET;

  // check by uint64_t element
  bool pass = true;
  size_t fail_cnt = 0;
  for (size_t i = 0; i < OFFSET; ++i) {
    auto src1 = rn50_backbone_bs9_data[off1 + i];
    auto src2 = qinfos_data[off2 + i];

    if (src1 != src2) {
      //   std::cout << "ERROR: [" << i << "], src1=" << src1 << ", src2=" <<
      //   src2
      //             << std::endl;
      printf("ERROR: [%d], src1=0x%lx, src2=0x%lx\n", (int)i, src1, src2);
      pass = false;
      fail_cnt++;
    } else {
      if (i < 256)
        printf("VALID: [%d], src1=0x%lx, src2=0x%lx\n", (int)i, src1, src2);
    }
  }

  if (!pass) {
    std::cout << "FAILED in data conversion!" << fail_cnt << "/" << OFFSET
              << "\n";
  } else {
    std::cout << "PASSED in data conversion for " << OFFSET << "!\n";
  }
  return 0;
}