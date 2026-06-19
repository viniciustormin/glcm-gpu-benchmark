/*
 * glcm_common.h — Tipos, constantes e funções compartilhadas entre
 *                 glcm_cpu.c, glcm_pipeline.c e benchmark.c
 */

#ifndef GLCM_COMMON_H
#define GLCM_COMMON_H

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Parâmetros do pipeline ────────────────────────────────────────────────── */
#define N_LEVELS    64       /* níveis de quantização (6 bits) */
#define PATCH_SIZE  64       /* lado do patch em pixels */
#define PATCH_STEP  32       /* passo de extração (50% overlap) */
#define N_ANGLES    4        /* ângulos: 0°, 45°, 90°, 135° */
#define N_PROPS     5        /* propriedades Haralick por ângulo */
#define N_FEATURES  (N_PROPS * N_ANGLES)   /* 20 features por patch */
#define N_CLUSTERS  3        /* clusters KMeans */
#define KMEANS_ITER 100      /* máximo de iterações KMeans */
#define KMEANS_INIT 5        /* reinicializações KMeans */
#define GLCM_ELEMS  (N_LEVELS * N_LEVELS)  /* 4096 entradas por GLCM */

/* ── Estrutura de resultado de um run ────────────────────────────────────── */
typedef struct {
    double total_time_s;
    double glcm_time_s;
    double clustering_time_s;
    double silhouette_score;
    int    n_patches;
} RunResult;

/* ── Timer portável (segundos) ───────────────────────────────────────────── */
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Leitura de imagem binária (.bin) ────────────────────────────────────── *
 * Formato: uint32 H, uint32 W, H*W valores uint16 row-major               *
 * Retorna buffer alocado; caller deve free(). Preenche *H e *W.            */
static inline uint16_t *load_image_bin(const char *path, int *H, int *W) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    uint32_t h, w;
    if (fread(&h, sizeof(uint32_t), 1, f) != 1 ||
        fread(&w, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Erro ao ler cabeçalho de %s\n", path);
        fclose(f); return NULL;
    }
    *H = (int)h; *W = (int)w;
    uint16_t *data = (uint16_t *)malloc((size_t)h * w * sizeof(uint16_t));
    if (!data) { fclose(f); return NULL; }
    fread(data, sizeof(uint16_t), (size_t)h * w, f);
    fclose(f);
    return data;
}

/* ── Quantização uint16 → uint8 (6 bits) via shift de 10 bits ───────────── */
static inline void quantize(const uint16_t *src, uint8_t *dst, int N) {
    for (int i = 0; i < N; i++)
        dst[i] = (uint8_t)(src[i] >> 10);
}

/* ── Cálculo do número de patches ───────────────────────────────────────── */
static inline int count_patches(int H, int W) {
    int ny = (H - PATCH_SIZE) / PATCH_STEP + 1;
    int nx = (W - PATCH_SIZE) / PATCH_STEP + 1;
    return ny * nx;
}

/* ── Extração de patches: preenche buffer contíguo [n_patches, PATCH_SIZE²] */
static inline void extract_patches(const uint8_t *img_q, int H, int W,
                                   uint8_t *patches_out, int *n_patches_out) {
    int p = 0;
    for (int y = 0; y <= H - PATCH_SIZE; y += PATCH_STEP) {
        for (int x = 0; x <= W - PATCH_SIZE; x += PATCH_STEP) {
            uint8_t *dst = patches_out + (size_t)p * PATCH_SIZE * PATCH_SIZE;
            for (int py = 0; py < PATCH_SIZE; py++)
                memcpy(dst + py * PATCH_SIZE,
                       img_q + (y + py) * W + x,
                       PATCH_SIZE);
            p++;
        }
    }
    *n_patches_out = p;
}

/* ── Propriedades Haralick a partir de uma GLCM normalizada ─────────────── *
 * glcm: float[N_LEVELS][N_LEVELS], já normalizado (soma = 1)               *
 * out[0..4]: contrast, dissimilarity, homogeneity, energy, correlation      */
static inline void haralick_props(const float *glcm, double *out) {
    double contrast = 0, dissim = 0, homog = 0, energy = 0;
    double mu_i = 0, mu_j = 0;

    for (int i = 0; i < N_LEVELS; i++) {
        for (int j = 0; j < N_LEVELS; j++) {
            double g = (double)glcm[i * N_LEVELS + j];
            double d = (double)(i - j);
            contrast += d * d * g;
            dissim   += fabs(d) * g;
            homog    += g / (1.0 + d * d);
            energy   += g * g;
            mu_i     += i * g;
            mu_j     += j * g;
        }
    }

    double sig_i = 1e-10, sig_j = 1e-10;
    for (int i = 0; i < N_LEVELS; i++) {
        for (int j = 0; j < N_LEVELS; j++) {
            double g = (double)glcm[i * N_LEVELS + j];
            sig_i += (i - mu_i) * (i - mu_i) * g;
            sig_j += (j - mu_j) * (j - mu_j) * g;
        }
    }
    sig_i = sqrt(sig_i);
    sig_j = sqrt(sig_j);

    double corr = 0.0;
    for (int i = 0; i < N_LEVELS; i++) {
        for (int j = 0; j < N_LEVELS; j++) {
            double g = (double)glcm[i * N_LEVELS + j];
            corr += (i - mu_i) * (j - mu_j) * g;
        }
    }
    corr /= (sig_i * sig_j);

    out[0] = contrast;
    out[1] = dissim;
    out[2] = homog;
    out[3] = energy;
    out[4] = corr;
}

/* ── KMeans simples (Lloyd, Euclidean) ───────────────────────────────────── *
 * features: [n, D] row-major double                                         *
 * labels_out: [n] int (output)                                              *
 * Retorna silhouette score.                                                  */
static inline double kmeans_and_silhouette(
    const double *features, int n, int D,
    int K, int max_iter, int n_init,
    int *labels_out)
{
    double *centroids = (double *)malloc(K * D * sizeof(double));
    double *best_cent = (double *)malloc(K * D * sizeof(double));
    int    *labels    = (int    *)malloc(n   * sizeof(int));
    double  best_inertia = 1e300;

    /* n_init reinicializações */
    for (int init = 0; init < n_init; init++) {
        /* Inicialização: escolher K pontos distintos aleatoriamente */
        /* Usamos índices espaçados deterministicamente para reprodutibilidade */
        for (int k = 0; k < K; k++) {
            int idx = (int)((long long)k * n / K);
            memcpy(centroids + k * D, features + (size_t)idx * D, D * sizeof(double));
        }

        for (int iter = 0; iter < max_iter; iter++) {
            /* Atribuição */
            int changed = 0;
            for (int i = 0; i < n; i++) {
                double best_d = 1e300;
                int    best_k = 0;
                const double *xi = features + (size_t)i * D;
                for (int k = 0; k < K; k++) {
                    double dist = 0.0;
                    const double *ck = centroids + k * D;
                    for (int d = 0; d < D; d++) {
                        double diff = xi[d] - ck[d];
                        dist += diff * diff;
                    }
                    if (dist < best_d) { best_d = dist; best_k = k; }
                }
                if (labels[i] != best_k) { labels[i] = best_k; changed++; }
            }
            if (iter > 0 && changed == 0) break;

            /* Atualização dos centróides */
            int *counts = (int *)calloc(K, sizeof(int));
            memset(centroids, 0, K * D * sizeof(double));
            for (int i = 0; i < n; i++) {
                int k = labels[i];
                counts[k]++;
                const double *xi = features + (size_t)i * D;
                double *ck = centroids + k * D;
                for (int d = 0; d < D; d++) ck[d] += xi[d];
            }
            for (int k = 0; k < K; k++) {
                if (counts[k] > 0) {
                    double *ck = centroids + k * D;
                    for (int d = 0; d < D; d++) ck[d] /= counts[k];
                }
            }
            free(counts);
        }

        /* Calcular inércia */
        double inertia = 0.0;
        for (int i = 0; i < n; i++) {
            const double *xi = features + (size_t)i * D;
            const double *ck = centroids + labels[i] * D;
            for (int d = 0; d < D; d++) {
                double diff = xi[d] - ck[d];
                inertia += diff * diff;
            }
        }
        if (inertia < best_inertia) {
            best_inertia = inertia;
            memcpy(best_cent, centroids, K * D * sizeof(double));
            memcpy(labels_out, labels, n * sizeof(int));
        }
    }

    /* ── Silhouette Score ─────────────────────────────────────────────────── */
    double sil_total = 0.0;
    /* Para eficiência, calculamos distância média intra e inter cluster */
    /* Usando amostragem se n > 1000 para manter tempo razoável */
    int sample = n > 1000 ? 1000 : n;
    int step   = n / sample;
    int counted = 0;

    for (int i = 0; i < n; i += step) {
        int ki = labels_out[i];
        const double *xi = features + (size_t)i * D;

        /* distância média intra-cluster (a) */
        double a = 0.0; int cnt_a = 0;
        for (int j = 0; j < n; j++) {
            if (j == i || labels_out[j] != ki) continue;
            double dist = 0.0;
            const double *xj = features + (size_t)j * D;
            for (int d = 0; d < D; d++) { double diff = xi[d]-xj[d]; dist += diff*diff; }
            a += sqrt(dist); cnt_a++;
        }
        if (cnt_a > 0) a /= cnt_a;

        /* distância média ao cluster mais próximo (b) */
        double b = 1e300;
        for (int k = 0; k < K; k++) {
            if (k == ki) continue;
            double bk = 0.0; int cnt_b = 0;
            for (int j = 0; j < n; j++) {
                if (labels_out[j] != k) continue;
                double dist = 0.0;
                const double *xj = features + (size_t)j * D;
                for (int d = 0; d < D; d++) { double diff = xi[d]-xj[d]; dist += diff*diff; }
                bk += sqrt(dist); cnt_b++;
            }
            if (cnt_b > 0) { bk /= cnt_b; if (bk < b) b = bk; }
        }

        double denom = a > b ? a : b;
        double si = denom > 1e-10 ? (b - a) / denom : 0.0;
        sil_total += si;
        counted++;
    }

    free(centroids); free(best_cent); free(labels);
    return counted > 0 ? sil_total / counted : 0.0;
}

/* ── Normalização Z-score (StandardScaler) ───────────────────────────────── *
 * Modifica features in-place. [n, D] row-major double.                      */
static inline void standard_scale(double *features, int n, int D) {
    for (int d = 0; d < D; d++) {
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < n; i++) mean += features[(size_t)i * D + d];
        mean /= n;
        for (int i = 0; i < n; i++) {
            double diff = features[(size_t)i * D + d] - mean;
            var += diff * diff;
        }
        var /= n;
        double std = sqrt(var + 1e-10);
        for (int i = 0; i < n; i++)
            features[(size_t)i * D + d] = (features[(size_t)i * D + d] - mean) / std;
    }
}

#endif /* GLCM_COMMON_H */
