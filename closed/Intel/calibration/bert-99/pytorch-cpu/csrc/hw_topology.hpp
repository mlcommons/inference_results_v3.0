#pragma once
#include <cstddef>
#include "kmp_launcher.hpp"

namespace hw {
//
// Hardware topology Description/Interface
//
class __hwtopo {
public:
  // TODO: Detect sockets and hyper-threading
  static __hwtopo* CreateHarewareTopology (size_t nSocket, bool bHT = true);
  virtual size_t AllocProc (int slot, int threadPerInstance) = 0;
};

}
