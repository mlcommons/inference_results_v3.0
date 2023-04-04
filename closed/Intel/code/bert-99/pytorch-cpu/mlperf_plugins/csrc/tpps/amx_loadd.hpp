#pragma once

#include <immintrin.h>

template <int dst> inline void __tile_loadd(void* src, int stride);

template <> inline void __tile_loadd<0>(void* src, int stride) { _tile_loadd(0, src, stride); }
template <> inline void __tile_loadd<1>(void* src, int stride) { _tile_loadd(1, src, stride); }
template <> inline void __tile_loadd<2>(void* src, int stride) { _tile_loadd(2, src, stride); }
template <> inline void __tile_loadd<3>(void* src, int stride) { _tile_loadd(3, src, stride); }
template <> inline void __tile_loadd<4>(void* src, int stride) { _tile_loadd(4, src, stride); }
template <> inline void __tile_loadd<5>(void* src, int stride) { _tile_loadd(5, src, stride); }
template <> inline void __tile_loadd<6>(void* src, int stride) { _tile_loadd(6, src, stride); }
template <> inline void __tile_loadd<7>(void* src, int stride) { _tile_loadd(7, src, stride); }
