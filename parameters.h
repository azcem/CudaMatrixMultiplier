#define TILE 32
#define TILE_WIDTH 32
#define N_ITER 100
#define bM 128
#define bN 128
#define bK 8
#define tM 8
#define tN 8
#define NUM_THREADS_PER_BLOCK ((bM / tM) * (bN / tN))
