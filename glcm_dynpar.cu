/*
 * glcm_dynpar.cu — CUDA Dynamic Parallelism para GLCM adaptativa
 *
 * Estratégia: o kernel pai analisa a heterogeneidade de cada patch
 * (desvio padrão dos pixels). Patches heterogêneos lançam sub-kernels
 * com mais threads; patches homogêneos usam configuração leve.
 *
 * Requer: sm_35+ para Dynamic Parallelism. RTX 4090 = sm_89 → ok.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 -rdc=true -shared -Xcompiler -fPIC \
 *        -o glcm_dynpar.so glcm_dynpar.cu \
 *        -lcudadevrt -lcudart_static -lrt -lpthread -ldl
 */

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N_LEVELS   64
#define GLCM_SIZE  (N_LEVELS * N_LEVELS)
#define BLOCK_HEAVY 256   /* patches heterogêneos */
#define BLOCK_LIGHT  64   /* patches homogêneos   */
/* Threshold de desvio padrão para classificar heterogeneidade */
#define STD_THRESHOLD 15.0f

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d — %s\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

/* -----------------------------------------------------------------------
 * Sub-kernel: processa um único ângulo de um patch.
 * Lançado dinamicamente pelo kernel pai.
 * --------------------------------------------------------------------- */
__global__ void glcm_child_kernel(
    const uint8_t* __restrict__ patch,
    int pH, int pW,
    int* g_glcm,
    int angle_idx
) {
    __shared__ int s_glcm[GLCM_SIZE];

    const int dx[4] = { 1,  1, 0, -1 };
    const int dy[4] = { 0,  1, 1,  1 };

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    for (int i = tid; i < GLCM_SIZE; i += bdim)
        s_glcm[i] = 0;
    __syncthreads();

    int n_pixels = pH * pW;
    for (int idx = tid; idx < n_pixels; idx += bdim) {
        int y = idx / pW, x = idx % pW;
        int nx = x + dx[angle_idx], ny = y + dy[angle_idx];
        if (nx >= 0 && nx < pW && ny >= 0 && ny < pH) {
            int v1 = patch[y * pW + x];
            int v2 = patch[ny * pW + nx];
            atomicAdd(&s_glcm[v1 * N_LEVELS + v2], 1);
            atomicAdd(&s_glcm[v2 * N_LEVELS + v1], 1);
        }
    }
    __syncthreads();

    /* merge condicional */
    for (int i = tid; i < GLCM_SIZE; i += bdim) {
        int val = s_glcm[i];
        if (val > 0)
            atomicAdd(&g_glcm[i], val);
    }
}

/* -----------------------------------------------------------------------
 * Kernel pai: um thread por patch.
 * Calcula std do patch e lança sub-kernels com tamanho adaptativo.
 * --------------------------------------------------------------------- */
__global__ void glcm_dynpar_parent(
    const uint8_t* __restrict__ patches,
    int n_patches, int pH, int pW,
    int* glcms
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_patches) return;

    const uint8_t* patch = patches + (size_t)p * pH * pW;
    int n_pixels = pH * pW;

    /* Calcular média e desvio padrão do patch */
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < n_pixels; i++) {
        float v = (float)patch[i];
        sum  += v;
        sum2 += v * v;
    }
    float mean = sum / n_pixels;
    float std  = sqrtf(fmaxf(sum2 / n_pixels - mean * mean, 0.0f));

    /* Escolher tamanho de bloco adaptativo */
    int child_block = (std > STD_THRESHOLD) ? BLOCK_HEAVY : BLOCK_LIGHT;

    for (int a = 0; a < 4; a++) {
        int* g_glcm = glcms + ((size_t)(p * 4 + a)) * GLCM_SIZE;
        glcm_child_kernel<<<1, child_block>>>(patch, pH, pW, g_glcm, a);
    }
    /* cudaDeviceSynchronize implícito ao retornar do pai */
}

extern "C" {
void compute_glcm_dynpar(
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

    /* Kernel pai: 1 thread por patch, blocos de 32 */
    int threads_parent = 32;
    int blocks_parent  = (n_patches + threads_parent - 1) / threads_parent;
    glcm_dynpar_parent<<<blocks_parent, threads_parent>>>(
        d_patches, n_patches, pH, pW, d_glcms);
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
