#include <stdio.h>

// a & b: input vectors
// c: output vector
// n: vector size
__global__ void CU_vector_add(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
