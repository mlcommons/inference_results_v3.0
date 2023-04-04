#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <asm/prctl.h>        /* Definition of ARCH_* constants */
#include <sys/syscall.h>      /* Definition of SYS_* constants */
#include <unistd.h>

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

namespace amx_init {

bool amx_init();

}