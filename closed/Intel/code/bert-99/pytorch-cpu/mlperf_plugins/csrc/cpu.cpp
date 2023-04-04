#include <dnnl.hpp>

namespace intel_mlperf {
dnnl::engine& g_cpu() {
  static dnnl::engine g_cpu_ (dnnl::engine::kind::cpu, 0);
  return g_cpu_;
}

}
