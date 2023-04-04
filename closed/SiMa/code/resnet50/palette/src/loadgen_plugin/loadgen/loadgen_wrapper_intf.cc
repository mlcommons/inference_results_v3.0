#include "loadgen_wrapper_intf.h"

LoadgenSingleton * LoadgenSingleton::loadgen(nullptr);
std::mutex LoadgenSingleton::m;

LoadgenSingleton *LoadgenSingleton::get_instance()
{
  std::unique_lock<std::mutex> lk(m);
  if (loadgen == nullptr) {
    std::cout << "--- MLPerf::Creator Creating loadgen object";
    loadgen = new LoadgenSingleton();
  }
  return loadgen;
};
