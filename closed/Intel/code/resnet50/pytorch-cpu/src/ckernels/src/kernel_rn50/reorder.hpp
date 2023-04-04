#pragma once
#include <stdint.h>


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 256x2048x7x7
**/
extern "C" bool reorder_256x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 256x64x56x56
**/
extern "C" bool reorder_256x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 4x2048x7x7
**/
extern "C" bool reorder_4x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 4x64x56x56
**/
extern "C" bool reorder_4x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 8x2048x7x7
**/
extern "C" bool reorder_8x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 8x64x56x56
**/
extern "C" bool reorder_8x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);