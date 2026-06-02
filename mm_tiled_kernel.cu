#include "clear.cu"
#include "loadTile.cu"
#include "mm.cu"
#include "writeTile.cu"

__global__ void mm_tiled_kernel(float *A, float *B, float *C, unsigned int M,
                                unsigned int N, unsigned int K) {

  // identify the block's tile
  unsigned int bRow = blockIdx.y * bM;
  unsigned int bCol = blockIdx.x * bN;

  // identify the thread's tile within the block
  unsigned int tilesPerBlockX = bN / tN;
  unsigned int ty = threadIdx.x / tilesPerBlockX;
  unsigned int tx = threadIdx.x % tilesPerBlockX;
  unsigned int tRow = ty * tM;
  unsigned int tCol = tx * tN;

  // initialize the output tile
  float C_r[tM][tN];
  clear(C_r, tM, tN);

  // iterate over the input tile
  __shared__ float Acurr_ss[bM * (bK + 1)];
  __shared__ float Anext_ss[bM * (bK + 1)];
  __shared__ float Bcurr_ss[bK * bN];
  __shared__ float Bnext_ss[bK * bN];

  float *Acurr_s = Acurr_ss;
  float *Anext_s = Anext_ss;
  float *Bcurr_s = Bcurr_ss;
  float *Bnext_s = Bnext_ss;

  // pre-feth first iteration tiles to shared memory
  loadTile(&A[bRow * K], K, M - bRow, K, &Acurr_s[0], bK + 1, bM, bK);
  loadTile(&B[bCol], N, K, N - bCol, &Bcurr_s[0], bN, bK, bN);
  __syncthreads();
  // condition is int-safe ceil of K/Bk
  for (unsigned int tile = 1; tile < (K + bK - 1) / bK; ++tile) {

    // compute with current iteration shared memory tiles
    mm(tM, tN, bK, &Acurr_s[tRow * (bK + 1)], bK + 1, &Bcurr_s[tCol], bN, C_r);

    // prefetch next iteration tiles to shared memory
    loadTile(&A[bRow * K + tile * bK], K, M - bRow, K - tile * bK, &Anext_s[0],
             bK + 1, bM, bK);
    loadTile(&B[tile * bK * N + bCol], N, K - tile * bK, N - bCol, &Bnext_s[0],
             bN, bK, bN);
    __syncthreads();

    // swap double buffers
    float *Atmp_s = Acurr_s;
    Acurr_s = Anext_s;
    Anext_s = Atmp_s;
    float *Btmp_s = Bcurr_s;
    Bcurr_s = Bnext_s;
    Bnext_s = Btmp_s;
  }
  // compute with last iteration shared memory tile
  mm(tM, tN, bK, &Acurr_s[tRow * (bK + 1)], bK + 1, &Bcurr_s[tCol], bN, C_r);

  // write output tile
  float *c = &C[(bRow + tRow) * N + bCol + tCol];
  unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
  unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
  writeTile(c, N, maxRow, maxCol, C_r, tM, tN);
}
