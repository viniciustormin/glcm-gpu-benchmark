/*
 * glcm_cuda.cu — Kernel CUDA básico para GLCM em lote de patches
 *
 * Estratégia: um bloco de threads por patch, cada thread processa múltiplos
 * pixels e faz atomicAdd diretamente na GLCM em memória global.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 -shared -fPIC -o glcm_cuda.so glcm_cuda.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#define N_LEVELS  64
#define BLOCK_DIM 256   /* threads por bloco = threads por patch */

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d — %s\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

/* -----------------------------------------------------------------------
 * Kernel: cada bloco processa um patch para um ângulo.
 * gridDim.x = n_patches   blockDim.x = BLOCK_DIM
 * Cada thread trata (pH*pW / BLOCK_DIM) pixels em loop.
 * --------------------------------------------------------------------- */
__global__ void glcm_basic_kernel(
    const uint8_t* __restrict__ patches,   /* [n_patches, pH, pW] */
    int n_patches, int pH, int pW,
    int* glcms,                            /* [n_patches, 4, N_LEVELS, N_LEVELS] */
    int angle_idx
) {
    const int dx[4] = { 1,  1, 0, -1 };
    const int dy[4] = { 0,  1, 1,  1 };

    int p   = blockIdx.x;
    int tid = threadIdx.x;
    int n_pixels = pH * pW;

    const uint8_t* patch = patches + (size_t)p * pH * pW;
    int* g_glcm = glcms + ((size_t)(p * 4 + angle_idx)) * N_LEVELS * N_LEVELS;

    for (int idx = tid; idx < n_pixels; idx += BLOCK_DIM) {
        int y = idx / pW;
        int x = idx % pW;

        int nx = x + dx[angle_idx];
        int ny = y + dy[angle_idx];

        if (nx >= 0 && nx < pW && ny >= 0 && ny < pH) {
            int v1 = patch[y * pW + x];
            int v2 = patch[ny * pW + nx];
            atomicAdd(&g_glcm[v1 * N_LEVELS + v2], 1);
            atomicAdd(&g_glcm[v2 * N_LEVELS + v1], 1);   /* simétrico */
        }
    }
}

/* -----------------------------------------------------------------------
 * Interface C exposta via ctypes.
 * Entradas  : patches uint8 [n_patches, pH, pW] (host)
 * Saída     : glcm_out float32 [n_patches, 4, N_LEVELS, N_LEVELS] (host)
 *             normalizado por ângulo (∑ = 1)
 * --------------------------------------------------------------------- */
extern "C" {
void compute_glcm_basic(
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
        glcm_basic_kernel<<<grid, block>>>(
            d_patches, n_patches, pH, pW, d_glcms, a);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copiar para host, normalizar por patch × ângulo */
    int* h_tmp = new int[glcm_elems];
    CUDA_CHECK(cudaMemcpy(h_tmp, d_glcms, glcm_bytes, cudaMemcpyDeviceToHost));

    int entry = N_LEVELS * N_LEVELS;
    for (int p = 0; p < n_patches; p++) {
        for (int a = 0; a < 4; a++) {
            const int*  src = h_tmp     + (p * 4 + a) * entry;
            float*      dst = h_glcm_out + (p * 4 + a) * entry;
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
