#include "hw_topology.hpp"

namespace hw {
template <int nSocket, int nHT>
class HardwareTopology : __hwtopo {
public:
  HardwareTopology () = default;

private:
  static const size_t nCores_;
  static const size_t nThreads_;

  static constexpr size_t nScoket_ = nSocket;
  static constexpr size_t nHT_ = nHT;
};

template <int nSocket, int nHT>
const size_t HardwareTopology<nSocket, nHT>::nCores_
      = kmp::KMPLauncher::getMaxProc() / nHT;

template <int nSocket, int nHT>
const size_t HardwareTopology<nSocket, nHT>::nThreads_
      = kmp::KMPLauncher::getMaxProc();
}
