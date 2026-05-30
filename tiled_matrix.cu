#include "matMultiplyTiledKernel.cu"
#include "parameters.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

bool areEqual(float a, float b) {
  float diff = fabsf(a - b);
  float tolerance = (1 / 100.0f) * fmax(fabsf(a), fabsf(b)) + 0.01;

  return diff <= tolerance;
}

bool matCompare(float *A, float *B, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    if (!areEqual(A[i], B[i])) {
      printf("%f, %f\n\n", A[i], B[i]);
      return false;
    }
  }
  return true;
}

void matPrint(float *A, int j, int k) {
  for (int i = 0; i < j; i++) {
    for (int m = 0; m < k; m++) {
      printf("%f ", A[i * k + m]);
    }
    printf("\n");
  }
}

void matPopulate(float *A, int j, int k) {
  srand(time(NULL));
  for (int i = 0; i < j * k; i++) {
    A[i] = rand() / 10000000.0;
  }
}

int main(int argc, char **argv) {
  int j;
  int k;
  int l;
  if (argc > 1) {
    j = atoi(argv[1]);
    k = atoi(argv[2]);
    l = atoi(argv[3]);
  } else {
    j = 3;
    k = 3;
    l = 3;
  }

  // allocate memory on CPU
  float *A = (float *)malloc(j * k * sizeof(float));
  float *B = (float *)malloc(k * l * sizeof(float));
  float *out_tiled = (float *)malloc(j * l * sizeof(float));
  float *out_cublas = (float *)malloc(j * l * sizeof(float));

  matPopulate(A, j, k);
  matPopulate(B, k, l);

  // allocate memory on GPU
  float *A_d;
  float *B_d;
  float *out_d;
  cudaMalloc((void **)&A_d, j * k * sizeof(float));
  cudaMalloc((void **)&B_d, k * l * sizeof(float));
  cudaMalloc((void **)&out_d, j * l * sizeof(float));

  // load CPU data to GPU
  cudaMemcpy(A_d, A, j * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, k * l * sizeof(float), cudaMemcpyHostToDevice);

  // call tiled matrix kernel
  /* TILED MATRIX KERNEL EXECUTION */
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  dim3 bd(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gd(ceil(l / (float)TILE_WIDTH), ceil(j / (float)TILE_WIDTH), 1);

  cudaEventRecord(start_t);
  for (int n = 0; n < N_ITER; n++)
    matMultiplyTiledKernel<<<gd, bd>>>(A_d, B_d, out_d, j, k, l);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  cudaEventRecord(stop_t);
  cudaEventSynchronize(stop_t);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_t, stop_t);
  printf("time taken in GPU (tiled): %f seconds\n", milliseconds / 1000.0);
  cudaMemcpy(out_tiled, out_d, j * l * sizeof(float), cudaMemcpyDeviceToHost);
  /* TILED GPU EXECUTION END */

  /* CUBLAS START */
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start_c, stop_c;
  cudaEventCreate(&start_c);
  cudaEventCreate(&stop_c);
  cudaEventRecord(start_c);

  for (int n = 0; n < N_ITER; n++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, l, j, k, &alpha, B_d, l, A_d,
                k, &beta, out_d, l);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  cudaEventRecord(stop_c);
  cudaEventSynchronize(stop_c);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_c, stop_c);
  printf("time taken in GPU (CUBLAS): %f seconds\n", milliseconds / 1000.0);
  cudaMemcpy(out_cublas, out_d, j * l * sizeof(float), cudaMemcpyDeviceToHost);
  /* CUBLAS END */

  // compare results
  if (matCompare(out_tiled, out_cublas, j, l)) {
    printf("two results are equal.\n");
  } else {
    printf("two results mismatch.\n");
  }

  // free CPU
  free(A);
  free(B);
  free(out_tiled);
  free(out_cublas);
  // free GPU
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(out_d);
}
