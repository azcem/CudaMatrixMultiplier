#include "matMultiplyTiledKernel.cu"
#include "parameters.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* multiply two sqaure matrices in CPU */
void matMultiply(float *A, float *B, float *out, int n) {
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      for (int k = 0; k < n; k++) {
        out[row * n + col] += A[row * n + k] * B[k * n + col];
      }
    }
  }
}

/* print sqaure matrix */
void matPrint(float *A, int n) {
  int counter = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f  ", A[counter]);
      counter++;
    }
    printf("\n");
  }
}

/** generate random sqaure matrix */
void matPopulate(float *A, int n) {
  int counter = 0;
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[counter] = rand() / 10000000.0;
      counter++;
    }
  }
}

/* GPU kernel to multiply sqaure matrices */
__global__ void matMultiplyKernel(float *A, float *B, float *out, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row >= n) || (col >= n)) {
    return;
  }
  float pValue = 0;
  for (int k = 0; k < n; k++) {
    pValue += A[row * n + k] * B[k * n + col];
  }
  out[row * n + col] = pValue;
}

bool areEqual(float a, float b) {
  float diff = fabsf(a - b);
  float tolerance = (1 / 100.0f) * fmax(fabsf(a), fabsf(b)) + 0.01;

  return diff <= tolerance;
}

/* compare two matrices are equal */
bool matCompareEqual(float *A, float *B, int n) {
  for (int i = 0; i < n * n; i++) {
    if (!areEqual(A[i], B[i])) {
      printf("%f, %f\n", A[i], B[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  int n;
  if (argc != 2)
    n = 3;
  else
    n = atoi(argv[1]);
  float *A = (float *)malloc(n * n * sizeof(float));
  float *B = (float *)malloc(n * n * sizeof(float));
  // float *out = (float *)malloc(n * n * sizeof(float));
  float *out_device = (float *)malloc(n * n * sizeof(float));
  float *out_cublas = (float *)malloc(n * n * sizeof(float));
  matPopulate(A, n);
  matPopulate(B, n);

  /* CPU EXECUTION */
  /*
  clock_t start, end;
  start = clock();
  matMultiply(A, B, out, n);
  end = clock();
  float cpu_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
  printf("time taken in CPU: %f seconds\n", cpu_time_used);
  */
  /* CPU EXECUTION END */

  // malloc memory in GPU
  float *A_d;
  float *B_d;
  float *out_d;
  cudaMalloc((void **)&A_d, n * n * sizeof(float));
  cudaMalloc((void **)&B_d, n * n * sizeof(float));
  cudaMalloc((void **)&out_d, n * n * sizeof(float));

  // move matrices data from CPU to GPU
  cudaMemcpy(A_d, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

  /* GPU EXECUTION START */
  /*
  // call kernel
  dim3 bd(TILE_WIDTH, TILE_WIDTH);
  dim3 gd(ceil(n / (TILE_WIDTH * 1.0)), ceil(n / (TILE_WIDTH * 1.0)), 1);
  cudaEvent_t start_d, stop_d;
  cudaEventCreate(&start_d);
  cudaEventCreate(&stop_d);

  cudaEventRecord(start_d);
  matMultiplyKernel<<<gd, bd>>>(A_d, B_d, out_d, n);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  cudaEventRecord(stop_d);
  cudaEventSynchronize(stop_d);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_d, stop_d);
  printf("time taken in GPU: %f seconds\n", milliseconds / 1000.0);
  //printf("speedup: %fx\n", cpu_time_used / (milliseconds / 1000.0));
  /* GPU EXECUTION END */
  // move GPU data back to CPU
  // cudaMemcpy(out_device, out_d, n * n * sizeof(float),
  // cudaMemcpyDeviceToHost);

  /* GPU TILED EXECUTION START */
  // call kernel
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  cudaEventRecord(start_t);
  matMultiplyTiledKernel<<<gd, bd>>>(A_d, B_d, out_d, n);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  cudaEventRecord(stop_t);
  cudaEventSynchronize(stop_t);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_t, stop_t);
  printf("time taken in GPU (tiled): %f seconds\n", milliseconds / 1000.0);
  // printf("speedup: %fx\n", cpu_time_used / (milliseconds / 1000.0));
  cudaMemcpy(out_device, out_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  /* GPU TILED EXECUTION END */

  /* CUBLAS implementation */
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start_c, stop_c;
  cudaEventCreate(&start_c);
  cudaEventCreate(&stop_c);
  cudaEventRecord(start_c);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A_d, n, B_d, n,
              &beta, out_d, n);

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
  // printf("speedup: %fx\n", cpu_time_used / (milliseconds / 1000.0));
  cudaMemcpy(out_cublas, out_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  /* CUBLAS implementation end*/

  if (matCompareEqual(out_cublas, out_device, n)) {
    printf("two results are equal.\n");
  } else {
    printf("two results mismatch.\n");

    matPrint(A, n);
    printf("\n\n\n");
    matPrint(B, n);
    printf("\n\n\n");
    matPrint(out_cublas, n);
    printf("\n\n\n");
    matPrint(out_device, n);
    printf("\n\n\n");
  }

  free(A);
  free(B);
  // free(out);
  free(out_device);
  free(out_cublas);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(out_d);
}
