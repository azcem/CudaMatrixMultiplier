A hand-tuned CUDA matrix multiplication kernel approaching cuBLAS performance on large square matrices, built from first principles. Implements four compounding optimizations: thread coarsening, register-tiled input caching, shared memory bank-conflict elimination, and software pipelining via double-buffering.

## Performance
It reaches 56.36% of cuBLAS performance on 4096x4096 matrices run on a Tesla T4 GPU.

## Optimizations
 
### 1. Thread Coarsening
 
Each thread computes a `tM × tN` output tile rather than a single element. This reduces the cost of loading from global memory as for each read from global memory its reused to compute more output elements, improving the compute-to-memory ratio.

### 2. Register Tiling of Input Tiles
 
Rather than re-indexing into shared memory on every inner-loop iteration, each thread caches its row of `A_s` and column of `B_s` into private register arrays before the dot product:
 
```cuda
float a_reg[tM];   // cache tM elements from A_s row
float b_reg[tN];   // cache tN elements from B_s col
 
for (int i = 0; i < bK; ++i) {
    for (int m = 0; m < tM; ++m) a_reg[m] = A_s[(tRow+m)*bK + i];
    for (int n = 0; n < tN; ++n) b_reg[n] = B_s[i*bN + tCol+n];
 
    for (int m = 0; m < tM; ++m)
        for (int n = 0; n < tN; ++n)
            C_reg[m][n] += a_reg[m] * b_reg[n];
}
```

### 3. Bank Conflict Elimination
 
A 32-bank shared memory architecture causes bank conflicts when multiple threads in a warp access the same bank simultaneously.
 
**Problem:** with `B_s[bK][bN]` laid out row-major, threads in a warp reading `B_s[i][tCol+n]` for consecutive `tCol` values stride across banks correctly — but reading down a column of `A_s` causes stride-`bK` accesses that collide.
 
**Fix:** pad the inner dimension of `A_s` by 1:
 
```cuda
__shared__ float A_s[bM][bK + 1];  // +1 padding eliminates bank conflicts
__shared__ float B_s[bK][bN];
```

### 4. Software Pipelining (Double Buffering)
 
Memory latency from global → shared loads can be hidden by overlapping the load for the *next* tile with computation on the *current* tile. Two shared memory buffers are allocated and swapped each iteration:

## Kernel Configuration
 
```cpp
#define bM  128    // block tile rows
#define bN  128    // block tile cols
#define bK  8    // block tile inner dim
#define tM   8    // thread tile rows
#define tN   8    // thread tile cols
 
// derived
#define NUM_THREADS_PER_BLOCK ((bM/tM) * (bN/tN))   // 256
```
 
**Launch:**
 
```cpp
dim3 blockDim(NUM_THREADS_PER_BLOCK, 1, 1);
dim3 gridDim((N + bN - 1) / bN, (M + bM - 1) / bM, 1);
 
mm_tiled_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);
```

## Build & Run
 
```bash
nvcc tiled_matrix.cu -o tiled_matrix -lcublas
./tiled_matrix 4096 4096 4096      # M K N
```

## References
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — Hwu, Kirk, Hajj — Ch. 5–6

