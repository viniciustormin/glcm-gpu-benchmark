/*
 * glcm_pipeline.c  —  Pipeline GPU via dlopen/dlsym.
 *
 * A biblioteca .so é aberta UMA VEZ e mantida aberta durante todo o
 * benchmark para evitar reinicialização do contexto CUDA a cada chamada.
 */

#define _POSIX_C_SOURCE 200809L
#include "glcm_common.h"
#include <dlfcn.h>

typedef void (*GlcmKernelFn)(const uint8_t *, int, int, int, float *);

/* Handle global — aberto uma vez, fechado ao final do programa */
static void          *g_lib     = NULL;
static GlcmKernelFn   g_kernel  = NULL;
static char           g_so_path[512] = {0};

/* Abre (ou reutiliza) a biblioteca e resolve o símbolo */
static int load_kernel(const char *so_path, const char *func_name) {
    /* Se já está aberta a mesma lib, só resolve o símbolo */
    if (g_lib && strcmp(g_so_path, so_path) == 0) {
        g_kernel = (GlcmKernelFn)dlsym(g_lib, func_name);
        return g_kernel ? 0 : -1;
    }
    /* Fechar lib anterior se diferente */
    if (g_lib) { dlclose(g_lib); g_lib = NULL; }

    g_lib = dlopen(so_path, RTLD_NOW);
    if (!g_lib) { fprintf(stderr, "dlopen(%s): %s\n", so_path, dlerror()); return -1; }
    snprintf(g_so_path, sizeof(g_so_path), "%s", so_path);

    g_kernel = (GlcmKernelFn)dlsym(g_lib, func_name);
    if (!g_kernel) { fprintf(stderr, "dlsym(%s): %s\n", func_name, dlerror()); return -1; }
    return 0;
}

/* Fecha a biblioteca (chamar no final do programa) */
void close_gpu_lib(void) {
    if (g_lib) { dlclose(g_lib); g_lib = NULL; }
}

RunResult run_gpu_pipeline(const char *image_path,
                            const char *so_path,
                            const char *func_name) {
    RunResult res = {0};

    if (load_kernel(so_path, func_name) != 0) return res;

    int H, W;
    uint16_t *img16 = load_image_bin(image_path, &H, &W);
    if (!img16) return res;

    double t0 = now_sec();

    int N = H * W;
    uint8_t *img8 = (uint8_t *)malloc(N);
    quantize(img16, img8, N);
    free(img16);

    int n_patches = count_patches(H, W);
    uint8_t *patches = (uint8_t *)malloc((size_t)n_patches * PATCH_SIZE * PATCH_SIZE);
    int actual = 0;
    extract_patches(img8, H, W, patches, &actual);
    free(img8);
    n_patches = actual;
    res.n_patches = n_patches;

    size_t glcm_elems = (size_t)n_patches * N_ANGLES * GLCM_ELEMS;
    float *glcm_out = (float *)calloc(glcm_elems, sizeof(float));

    double t1 = now_sec();

    g_kernel(patches, n_patches, PATCH_SIZE, PATCH_SIZE, glcm_out);
    free(patches);

    double *features = (double *)malloc((size_t)n_patches * N_FEATURES * sizeof(double));
    for (int p = 0; p < n_patches; p++) {
        for (int a = 0; a < N_ANGLES; a++) {
            const float *glcm = glcm_out + ((size_t)p * N_ANGLES + a) * GLCM_ELEMS;
            double props[N_PROPS];
            haralick_props(glcm, props);
            for (int k = 0; k < N_PROPS; k++)
                features[(size_t)p * N_FEATURES + a * N_PROPS + k] = props[k];
        }
    }
    free(glcm_out);

    double t2 = now_sec();

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

#ifdef STANDALONE_MAIN
int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,
            "Uso: %s <imagem.bin> <kernel.so> <func_name>\n", argv[0]);
        return 1;
    }
    RunResult r = run_gpu_pipeline(argv[1], argv[2], argv[3]);
    printf("GPU pipeline [%s] — %s\n", argv[3], argv[1]);
    printf("  total_time_s     : %.4f\n", r.total_time_s);
    printf("  glcm_time_s      : %.4f\n", r.glcm_time_s);
    printf("  clustering_time_s: %.4f\n", r.clustering_time_s);
    printf("  silhouette_score : %.4f\n", r.silhouette_score);
    printf("  n_patches        : %d\n",   r.n_patches);
    close_gpu_lib();
    return 0;
}
#endif /* STANDALONE_MAIN */
