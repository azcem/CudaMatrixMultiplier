__device__ __forceinline__ void loadTile(float *T, unsigned int lda,
                                         unsigned int maxRow,
                                         unsigned int maxCol, float *T_s,
                                         unsigned int ldas, unsigned int height,
                                         unsigned int width) {
  unsigned int rowsPerSubtile = NUM_THREADS_PER_BLOCK / width;
  unsigned int numSubtiles = height / rowsPerSubtile;

#pragma unroll
  for (unsigned int subTile = 0; subTile < numSubtiles; ++subTile) {
    unsigned int row = subTile * rowsPerSubtile + threadIdx.x / width;
    unsigned int col = threadIdx.x % width;
    if (row < maxRow && col < maxCol) {
      T_s[row * ldas + col] = T[row * lda + col];
    } else {
      T_s[row * ldas + col] = 0.0f;
    }
  }
}
