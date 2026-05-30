#pragma once
#include "parameters.h"

/* GPU kernel to perform tiled matrix multiplication */
// j -> height of A or height of P
// k -> width of A or height of B
// l -> width of B or width of P
__global__ void matMultiplyTiledKernel(float *A, float *B, float *P, int j,
                                       int k, int l) {
  // calulcate row and col indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  // declare shared variables
  __shared__ float Adx[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bdx[TILE_WIDTH][TILE_WIDTH];
  // loop over phases
  float pValue = 0;
  for (int ph = 0; ph < ceil(k / (float)TILE_WIDTH); ++ph) {
    // for each phase load one element from A and B
    if ((row < j) && (ph * TILE_WIDTH + tx) < k)
      Adx[ty][tx] = A[row * k + ph * TILE_WIDTH + tx];
    else
      Adx[ty][tx] = 0.0f;

    if (col < l && (ph * TILE_WIDTH + ty) < k)
      Bdx[ty][tx] = B[ph * TILE_WIDTH * l + ty * l + col];
    else
      Bdx[ty][tx] = 0.0f;
    // sync threads
    __syncthreads();
    // loop over Adx and Bdx and calculate pValue
    for (int m = 0; m < TILE_WIDTH; ++m) {
      pValue += Adx[ty][m] * Bdx[m][tx];
    }
    // sync threads
    __syncthreads();
  }
  // load pValue into P
  if (row < j && col < l)
    P[row * l + col] = pValue;
}
