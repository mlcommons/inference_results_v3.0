#pragma once
#include <stdint.h>


/**
 * Do reorder from AB format to BA16a64b2a format. Plain dims is 13x512
**/
extern "C" bool reorder_13x512_AB_BA16a64b2a(uint16_t* out, uint16_t* in);


/**
 * Do reorder from BA16a64b2a format to AB format. Plain dims is 13x512
**/
extern "C" bool reorder_13x512_BA16a64b2a_AB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from AB format to BA64a64b2a format. Plain dims is 256x128
**/
extern "C" bool reorder_256x128_AB_BA64a64b2a(uint16_t* out, uint16_t* in);


/**
 * Do reorder from BA64a64b2a format to AB format. Plain dims is 256x128
**/
extern "C" bool reorder_256x128_BA64a64b2a_AB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from AB format to BA64a64b2a format. Plain dims is 512x256
**/
extern "C" bool reorder_512x256_AB_BA64a64b2a(uint16_t* out, uint16_t* in);


/**
 * Do reorder from BA64a64b2a format to AB format. Plain dims is 512x256
**/
extern "C" bool reorder_512x256_BA64a64b2a_AB(uint16_t* out, uint16_t* in);
