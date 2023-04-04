#pragma once

#include <iostream>

namespace intel_mlperf
{

void print_int8_2Dmatrix(const int8_t *ptr, int row, int col, int stride);
void print_int32_2Dmatrix(const int *ptr, int row, int col, int stride);

void print_zero_pos_int32(const int *ptr, int row, int col, int stride);
void print_zero_pos_int8(const int8_t *ptr, int row, int col, int stride);

template <class T>
void compare_matrix(const T *a, const T *b, int row, int col, int lda, int ldb);

void compare_naive_input(int* a, int8_t* b, int row, int col, int lda, int ldb);
void compare_naive_weight(int* a, int8_t* b, int row, int col, int lda, int ldb);
void compare_naive_output(int* a, int8_t* b, int row, int col, int lda, int ldb);

template <class T>
void print_2d_matrix(const T *ptr, int row, int col, int stride)
{
  auto p = reinterpret_cast<const T(*)[stride]>(ptr);
  printf("---------------print 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%d\t", static_cast<const int>(p[i][j]));
    }
    printf("\n");
  }
  printf("---------------print 2d matrix------------------\n");
}

template <class T>
void compare_matrix(const T *a, const T *b, int row, int col, int lda, int ldb)
{
  auto a_ = reinterpret_cast<const T(*)[lda]>(a);
  auto b_ = reinterpret_cast<const T(*)[ldb]>(b);
  printf("---------------compare 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      auto re = a_[i][j] - b_[i][j];
      if (re != 0)
      {
        printf("(%d, %d) %d <--> %d\n", i, j, a_[i][j], b_[i][j]);
      }
    }
  }
  printf("---------------compare 2d matrix end------------------\n");
}

}