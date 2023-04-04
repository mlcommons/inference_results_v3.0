#include <immintrin.h>

template <int src> inline void __tile_stored(void* dst, int stride);

template <> inline void __tile_stored<0>(void* dst, int stride) { _tile_stored(0, dst, stride); }
template <> inline void __tile_stored<1>(void* dst, int stride) { _tile_stored(1, dst, stride); }
template <> inline void __tile_stored<2>(void* dst, int stride) { _tile_stored(2, dst, stride); }
template <> inline void __tile_stored<3>(void* dst, int stride) { _tile_stored(3, dst, stride); }
template <> inline void __tile_stored<4>(void* dst, int stride) { _tile_stored(4, dst, stride); }
template <> inline void __tile_stored<5>(void* dst, int stride) { _tile_stored(5, dst, stride); }
template <> inline void __tile_stored<6>(void* dst, int stride) { _tile_stored(6, dst, stride); }
template <> inline void __tile_stored<7>(void* dst, int stride) { _tile_stored(7, dst, stride); }
