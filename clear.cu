__device__ __forceinline__ void clear(float C_r[][tN], unsigned int m,
                                      unsigned int n) {
#pragma unroll
  for (unsigned int row = 0; row < m; ++row) {
    for (unsigned int col = 0; col < n; ++col) {
      C_r[row][col] = 0.0f;
    }
  }
}
