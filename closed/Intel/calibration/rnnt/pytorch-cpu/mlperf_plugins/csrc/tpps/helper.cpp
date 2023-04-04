#include "helper.hpp"
#include <stdlib.h>

namespace intel_mlperf
{

void print_int8_2Dmatrix(const int8_t *ptr, int row, int col, int stride)
{
  auto p = reinterpret_cast<const int8_t(*)[stride]>(ptr);
  printf("---------------print int8 2D matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%d\t", static_cast<int>(p[i][j]));
    }
    printf("\n");
  }
  printf("---------------print int8 2D matrix------------------\n");
}

void print_int32_2Dmatrix(const int *ptr, int row, int col, int stride)
{
  auto p = reinterpret_cast<const int(*)[stride]>(ptr);
  printf("---------------print int32 2D matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%d\t", p[i][j]);
    }
    printf("\n");
  }
  printf("---------------print int32 2D matrix------------------\n");
}

void print_zero_pos_int32(const int *ptr, int row, int col, int stride)
{
  printf("---------------print int32 zero pos------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      if (static_cast<int>(ptr[i * stride + j]) == 0)
      {
        printf("[%d, %d]\n", i, j);
      }
    }
  }
  printf("---------------print int32 zero pos end------------------\n");
}

void print_zero_pos_int8(const int8_t *ptr, int row, int col, int stride)
{
  printf("---------------print int8 zero pos------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      if (ptr[i * stride + j] == 0)
      {
        printf("[%d, %d]\n", i, j);
      }
    }
  }
  printf("---------------print int8 zero pos end------------------\n");
}

void set_data_act(void *a, size_t n_tile)
{
  srand(1);
  auto a_ = reinterpret_cast<int8_t (*)>(a);
  size_t elenum = n_tile * 16 * 1024;
  for (int i = 0; i < elenum; i++) {
    a_[i] = (rand() % 0xff);
  }
}

void set_data_wei(void *w, void* b) {
  srand(2);
  auto w_ = reinterpret_cast<int8_t (*)>(w);
  size_t elenum = 256 * 256;
  for (int i = 0; i < elenum; i++) {
    w_[i] = (rand() % 0xff);
  }
  auto b_ = reinterpret_cast<float (*)>(b);
  for (int i = 0; i < 256; i++) {
    b_[i] = rand();
  }
}

template <>
void print_2d_matrix(const uint8_t *ptr, int row, int col, int stride)
{
  auto p = reinterpret_cast<const uint8_t(*)[stride]>(ptr);
  printf("---------------print 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%d\t", static_cast<unsigned>(p[i][j]));
    }
    printf("\n");
  }
  printf("---------------print 2d matrix------------------\n");
}

template <>
void print_2d_matrix(const float *ptr, int row, int col, int stride)
{
  auto p = reinterpret_cast<const float(*)[stride]>(ptr);
  printf("---------------print 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%.0f\t", (p[i][j]));
    }
    printf("\n");
  }
  printf("---------------print 2d matrix------------------\n");
}

void compare_naive_input(int* a, int8_t* b, int row, int col, int lda, int ldb) {
  auto a_ = reinterpret_cast<int (*)[lda]>(a);
  auto b_ = reinterpret_cast<int8_t (*)[ldb]>(b);
  printf("---------------compare 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      auto re = static_cast<int8_t>(a_[i][j]) - b_[i][j];
      if (re != 0)
      {
        printf("(%d, %d) %d <--> %d\n", i, j, a_[i][j], b_[i][j]);
      }
    }
  }
  printf("---------------compare 2d matrix end------------------\n");
}

void compare_naive_weight(int* a, int8_t* b, int row, int col, int lda, int ldb) {
  auto a_ = reinterpret_cast<int (*)[lda]>(a);
  auto b_ = reinterpret_cast<int8_t (*)[ldb]>(b);
  printf("---------------compare 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      auto re = b_[i][j] - static_cast<int8_t>(a_[i * 4 + j % 4][j / 4]);
      if (re != 0)
      {
        printf("(%d, %d) %d <--> %d\n", i, j, a_[i][j], b_[i][j]);
      }
    }
  }
  printf("---------------compare 2d matrix end------------------\n");
}

void compare_naive_output(int* a, int8_t* b, int row, int col, int lda, int ldb) {
  auto a_ = reinterpret_cast<int (*)[lda]>(a);
  auto b_ = reinterpret_cast<int8_t (*)[ldb]>(b);
  printf("---------------compare 2d matrix------------------\n");
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      auto re = b_[i][j] - static_cast<int8_t>(a_[i][j]);
      if (re != 0)
      {
        printf("(%d, %d) %d <--> %d\n", i, j, a_[i][j], b_[i][j]);
      }
    }
  }
  printf("---------------compare 2d matrix end------------------\n");
}

}