#include <stdio.h>

__global__ void matrix_mult(int* a, int* b, int* c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    c[row * n + col] = 0;
    for (int k = 0; k < n; k++) {
      c[row * n + col] += a[row * n + k] * b[k * n + col];
    }
  }
}


int main() {
  int n = 1 << 12;
  int bytes = n * n * sizeof(int);
  
  int* h_a, *h_b, *h_c;
  int* d_a, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      h_a[i*n + j] = j*i;
      h_b[i*n + j] = j*i -2;
    }
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

  // threads per block
  int BLOCK_SIZE = 16;

  // blocks in each dimensions
  int GRID_SIZE = (int) ceil(n / BLOCK_SIZE) + 1;

  dim3 grid(GRID_SIZE, GRID_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  matrix_mult <<<grid, threads>>> (d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  printf("Done\n");

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}