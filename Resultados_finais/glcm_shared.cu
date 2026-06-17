/*
 * glcm_shared.cu — Kernel CUDA otimizado com shared memory para GLCM em lote
 *
 * Melhoria sobre glcm_cuda.cu: cada bloco acumula sua contribuição numa GLCM
 * privada em shared memory (atomicAdd na SHMEM é ~10-30x mais rápido que na
 * memória global). Ao final, o bloco faz um merge atômico para a GLCM global.
 *
 * Análise de memória por bloco:
 *   GLCM_SIZE = 64 × 64 = 4096 entradas × 4 bytes = 16 KB
 *   Limite H100 (sm_89): 228 KB por bloco → seguro.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 -shared -fPIC -o glcm_shared.so glcm_shared.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#define N_LEVELS  64
#define GLCM_SIZE (N_LEVELS * N_LEVELS)   /* 4096 */
#define BLOCK_DIM 256

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d — %s\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

/* -----------------------------------------------------------------------
 * Kernel: um bloco por patch.
 *
 * Fase 1 — zerar GLCM local na shared memory.
 * Fase 2 — cada thread itera sobre seus pixels e acumula em s_glcm (rápido).
 * Fase 3 — merge s_glcm → g_glcm com atomicAdd (um acesso global por entrada,
 *           em vez de um por pixel-par).
 * --------------------------------------------------------------------- */
__global__ void glcm_shared_kernel(
    const uint8_t* __restrict__ patches,   /* [n_patches, pH, pW] */
    int n_patches, int pH, int pW,
    int* glcms,                            /* [n_patches, 4, N_LEVELS, N_LEVELS] */
    int angle_idx
) {
    __shared__ int s_glcm[GLCM_SIZE];     /* 16 KB por bloco */

    const int dx[4] = { 1,  1, 0, -1 };
    const int dy[4] = { 0,  1, 1,  1 };

    int p   = blockIdx.x;
    int tid = threadIdx.x;

    /* ── Fase 1: zerar GLCM local ─────────────────────────────────────── */
    for (int i = tid; i < GLCM_SIZE; i += BLOCK_DIM)
        s_glcm[i] = 0;
    __syncthreads();

    /* ── Fase 2: acumular em shared memory ────────────────────────────── */
    if (p < n_patches) {
        const uint8_t* patch = patches + (size_t)p * pH * pW;
        int n_pixels = pH * pW;

        for (int idx = tid; idx < n_pixels; idx += BLOCK_DIM) {
            int y = idx / pW;
            int x = idx % pW;

            int nx = x + dx[angle_idx];
            int ny = y + dy[angle_idx];

            if (nx >= 0 && nx < pW && ny >= 0 && ny < pH) {
                int v1 = patch[y * pW + x];
                int v2 = patch[ny * pW + nx];
                atomicAdd(&s_glcm[v1 * N_LEVELS + v2], 1);
                atomicAdd(&s_glcm[v2 * N_LEVELS + v1], 1);
            }
        }
    }
    __syncthreads();

    /* ── Fase 3: merge shared → global ───────────────────────────────── */
    if (p < n_patches) {
        int* g_glcm = glcms + ((size_t)(p * 4 + angle_idx)) * GLCM_SIZE;
        for (int i = tid; i < GLCM_SIZE; i += BLOCK_DIM)
            atomicAdd(&g_glcm[i], s_glcm[i]);
    }
}

/* -----------------------------------------------------------------------
 * Interface C idêntica à do glcm_cuda.cu — substituição direta.
 * --------------------------------------------------------------------- */
extern "C" {
void compute_glcm_shared(
    const uint8_t* h_patches,
    int n_patches, int pH, int pW,
    float* h_glcm_out
) {
    size_t patches_bytes = (size_t)n_patches * pH * pW * sizeof(uint8_t);
    size_t glcm_elems    = (size_t)n_patches * 4 * N_LEVELS * N_LEVELS;
    size_t glcm_bytes    = glcm_elems * sizeof(int);

    uint8_t* d_patches;
    int*     d_glcms;

    CUDA_CHECK(cudaMalloc(&d_patches, patches_bytes));
    CUDA_CHECK(cudaMalloc(&d_glcms,   glcm_bytes));
    CUDA_CHECK(cudaMemcpy(d_patches, h_patches, patches_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_glcms, 0, glcm_bytes));

    dim3 grid(n_patches);
    dim3 block(BLOCK_DIM);

    for (int a = 0; a < 4; a++) {
        glcm_shared_kernel<<<grid, block>>>(
            d_patches, n_patches, pH, pW, d_glcms, a);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_tmp = new int[glcm_elems];
    CUDA_CHECK(cudaMemcpy(h_tmp, d_glcms, glcm_bytes, cudaMemcpyDeviceToHost));

    int entry = N_LEVELS * N_LEVELS;
    for (int p = 0; p < n_patches; p++) {
        for (int a = 0; a < 4; a++) {
            const int* src = h_tmp      + (p * 4 + a) * entry;
            float*     dst = h_glcm_out + (p * 4 + a) * entry;
            long long sum = 0;
            for (int i = 0; i < entry; i++) sum += src[i];
            float denom = sum > 0 ? (float)sum : 1.0f;
            for (int i = 0; i < entry; i++) dst[i] = src[i] / denom;
        }
    }

    delete[] h_tmp;
    CUDA_CHECK(cudaFree(d_patches));
    CUDA_CHECK(cudaFree(d_glcms));
}
} /* extern "C" */
