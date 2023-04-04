#include "postprocessor.hpp"

namespace rebel {

void IdentityFunc(const void* const* result_from_model, void** output) {
  auto result_from_model_casted = const_cast<void**>(result_from_model);
  *output = reinterpret_cast<void*>(result_from_model_casted[0]);
}

}  // namespace rebel
