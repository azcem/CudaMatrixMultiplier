__device__ __forceinline__ void mm(unsigned int m, unsigned int n,
                                   unsigned int k, float *a, unsigned int lda,
                                   float *b, unsigned int ldb, float c[][tN]) {

  float a_r[tM];
  float b_r[tN];
#pragma unroll
  for (unsigned int i = 0; i < k; ++i) {

// load A strip to registers
#pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
      a_r[row] = a[row * lda + i];
    }

// load B strip to registers
#pragma unroll
    for (unsigned int col = 0; col < n; ++col) {
      b_r[col] = b[i * ldb + col];
    }

// compute with strips
#pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
#pragma unroll
      for (unsigned int col = 0; col < n; ++col) {
        c[row][col] += a_r[row] * b_r[col];
      }
    }
  }
}
