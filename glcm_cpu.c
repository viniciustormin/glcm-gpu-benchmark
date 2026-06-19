/*
 * glcm_cpu.c  —  Pipeline CPU sequencial: baseline para benchmark.
 *
 * Substitui glcm_cpu.py. Implementa em C puro (sem Python/numpy/skimage):
 *   1. Leitura da imagem binária
 *   2. Quantização uint16 → uint8 (6 bits)
 *   3. Extração de patches
 *   4. Cálculo de GLCM (4 ângulos, simétrica, normalizada)
 *   5. Extração de 5 propriedades de Haralick por ângulo (20 features/patch)
 *   6. Normalização Z-score
 *   7. KMeans (K=3) + Silhouette Score
 *
 * Compile:
 *   gcc -O3 -o glcm_cpu glcm_cpu.c -lm
 *
 * Uso:
 *   ./glcm_cpu ./data/image_512x512.bin
 */

#include "glcm_common.h"

/* ── Ângulos: deslocamentos (dx, dy) para 0°, 45°, 90°, 135° ────────────── */
static const int DX[N_ANGLES] = { 1,  1, 0, -1 };
static const int DY[N_ANGLES] = { 0,  1, 1,  1 };

/* ── Cálculo de GLCM para um patch e um ângulo ──────────────────────────── *
 * glcm_out: float[N_LEVELS * N_LEVELS] zerado antes de chamar             */
static void compute_glcm_patch(const uint8_t *patch, int pH, int pW,
                                int angle_idx, float *glcm_out) {
    int dx = DX[angle_idx], dy = DY[angle_idx];
    long long total = 0;

    for (int y = 0; y < pH; y++) {
        for (int x = 0; x < pW; x++) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= pW || ny < 0 || ny >= pH) continue;
            int v1 = patch[y * pW + x];
            int v2 = patch[ny * pW + nx];
            glcm_out[v1 * N_LEVELS + v2] += 1.0f;
            glcm_out[v2 * N_LEVELS + v1] += 1.0f;   /* simétrica */
            total += 2;
        }
    }

    /* Normalizar */
    if (total > 0) {
        float inv = 1.0f / (float)total;
        for (int i = 0; i < GLCM_ELEMS; i++)
            glcm_out[i] *= inv;
    }
}

/* ── Pipeline CPU completo ──────────────────────────────────────────────── */
RunResult run_cpu_pipeline(const char *image_path) {
    RunResult res = {0};

    int H, W;
    uint16_t *img16 = load_image_bin(image_path, &H, &W);
    if (!img16) { fprintf(stderr, "Falha ao carregar %s\n", image_path); return res; }

    double t0 = now_sec();

    /* ── Quantização ─────────────────────────────────────────────────────── */
    int N = H * W;
    uint8_t *img8 = (uint8_t *)malloc(N * sizeof(uint8_t));
    quantize(img16, img8, N);
    free(img16);

    /* ── Extração de patches ─────────────────────────────────────────────── */
    int n_patches = count_patches(H, W);
    uint8_t *patches = (uint8_t *)malloc((size_t)n_patches * PATCH_SIZE * PATCH_SIZE);
    int actual_patches = 0;
    extract_patches(img8, H, W, patches, &actual_patches);
    free(img8);
    n_patches = actual_patches;
    res.n_patches = n_patches;

    double t1 = now_sec();

    /* ── GLCM + Haralick ─────────────────────────────────────────────────── */
    double *features = (double *)malloc((size_t)n_patches * N_FEATURES * sizeof(double));
    float  *glcm     = (float  *)malloc(GLCM_ELEMS * sizeof(float));

    for (int p = 0; p < n_patches; p++) {
        const uint8_t *patch = patches + (size_t)p * PATCH_SIZE * PATCH_SIZE;
        for (int a = 0; a < N_ANGLES; a++) {
            memset(glcm, 0, GLCM_ELEMS * sizeof(float));
            compute_glcm_patch(patch, PATCH_SIZE, PATCH_SIZE, a, glcm);
            double props[N_PROPS];
            haralick_props(glcm, props);
            for (int k = 0; k < N_PROPS; k++)
                features[(size_t)p * N_FEATURES + a * N_PROPS + k] = props[k];
        }
    }

    free(glcm);
    free(patches);

    double t2 = now_sec();

    /* ── Normalização + KMeans + Silhouette ─────────────────────────────── */
    standard_scale(features, n_patches, N_FEATURES);

    int *labels = (int *)malloc(n_patches * sizeof(int));
    double sil = kmeans_and_silhouette(features, n_patches, N_FEATURES,
                                        N_CLUSTERS, KMEANS_ITER, KMEANS_INIT,
                                        labels);
    free(labels);
    free(features);

    double t3 = now_sec();

    res.total_time_s      = t3 - t0;
    res.glcm_time_s       = t2 - t1;
    res.clustering_time_s = t3 - t2;
    res.silhouette_score  = sil;
    return res;
}

/* ── Main (uso standalone) ──────────────────────────────────────────────── */
#ifdef STANDALONE_MAIN
int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "./data/image_512x512.bin";
    printf("CPU pipeline — %s\n", path);
    RunResult r = run_cpu_pipeline(path);
    printf("  total_time_s     : %.4f\n", r.total_time_s);
    printf("  glcm_time_s      : %.4f\n", r.glcm_time_s);
    printf("  clustering_time_s: %.4f\n", r.clustering_time_s);
    printf("  silhouette_score : %.4f\n", r.silhouette_score);
    printf("  n_patches        : %d\n",   r.n_patches);
    return 0;
}
#endif /* STANDALONE_MAIN */
