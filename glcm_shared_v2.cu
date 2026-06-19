/*
 * glcm_shared_v2.cu — Shared Memory com merge CONDICIONAL (pula zeros)
 *
 * Fix do problema identificado nos resultados preliminares:
 *   Merge incondicional percorria todas as 4096 entradas da GLCM,
 *   incluindo zeros, gerando ~921K atomicAdds extras no 512².
 *
 * Solução: na Fase 3, verificar s_glcm[i] > 0 antes do atomicAdd global.
 *   Reduz atomicAdds globais em ~60-80% dependendo da esparsidade da GLCM.
 *
 * Compile (RTX 4090 / sm_89):
 *   nvcc -O3 -arch=sm_89 -shared -Xcompiler -fPIC -o glcm_shared_v2.so glcm_shared_v2.cu
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
 * Kernel v2: merge CONDICIONAL — só escreve entradas não-zero.
 * --------------------------------------------------------------------- */
__global__ void glcm_shared_v2_kernel(
    const uint8_t* __restrict__ patches,
    int n_patches, int pH, int pW,
    int* glcms,
    int angle_idx
) {
    __shared__ int s_glcm[GLCM_SIZE];   /* 16 KB */

    const int dx[4] = { 1,  1, 0, -1 };
    const int dy[4] = { 0,  1, 1,  1 };

    int p   = blockIdx.x;
    int tid = threadIdx.x;

    /* Fase 1: zerar GLCM local */
    for (int i = tid; i < GLCM_SIZE; i += BLOCK_DIM)
        s_glcm[i] = 0;
    __syncthreads();

    /* Fase 2: acumular em shared memory */
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

    /* Fase 3: merge CONDICIONAL — pula entradas zero */
    if (p < n_patches) {
        int* g_glcm = glcms + ((size_t)(p * 4 + angle_idx)) * GLCM_SIZE;
        for (int i = tid; i < GLCM_SIZE; i += BLOCK_DIM) {
            int val = s_glcm[i];
            if (val > 0)   /* ← fix principal */
                atomicAdd(&g_glcm[i], val);
        }
    }
}

extern "C" {
void compute_glcm_shared_v2(
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
        glcm_shared_v2_kernel<<<grid, block>>>(
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
