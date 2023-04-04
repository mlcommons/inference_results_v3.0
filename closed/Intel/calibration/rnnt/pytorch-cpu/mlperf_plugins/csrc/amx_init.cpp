#include "amx_init.hpp"

namespace amx_init {

bool amx_init() {
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

}