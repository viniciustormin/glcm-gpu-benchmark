/*
 * glcm_data.c  —  Gerador de imagens sintéticas uint16 com texturas variadas.
 *
 * Substitui glcm_data.py. Gera três regiões texturais inspiradas no paper CHASM:
 *   - Fundo suave (tecido normal)
 *   - Região tumoral homogênea (círculo central)
 *   - Região tumoral heterogênea (deslocada)
 *   - Região de textura intermediária (terceiro quadrante)
 *
 * Salva cada imagem em formato binário simples:
 *   ./data/image_{SIZE}x{SIZE}.bin
 *   Cabeçalho: uint32 height, uint32 width
 *   Dados:     height*width valores uint16, row-major
 *
 * Compile:
 *   gcc -O2 -o glcm_data glcm_data.c -lm
 *
 * Uso:
 *   ./glcm_data
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

/* ── Configuração ─────────────────────────────────────────────────────────── */
static const int SIZES[]  = {256, 512, 1024};
static const int N_SIZES  = 3;
static const uint64_t SEED = 42;

/* ── LCG: gerador de números aleatórios simples (substitui numpy.random) ─── */
typedef struct { uint64_t state; } RNG;

static void rng_init(RNG *r, uint64_t seed) { r->state = seed ^ 0x853c49e6748fea9bULL; }

/* Retorna double uniforme em [0, 1) */
static double rng_uniform(RNG *r) {
    r->state ^= r->state >> 12;
    r->state ^= r->state << 25;
    r->state ^= r->state >> 27;
    uint64_t v = r->state * 0x2545f4914f6cdd1dULL;
    return (double)(v >> 11) / (double)(1ULL << 53);
}

/* Retorna double uniforme em [lo, hi) */
static double rng_range(RNG *r, double lo, double hi) {
    return lo + rng_uniform(r) * (hi - lo);
}

/* ── Filtro Gaussiano 2D (separável, borda zero-pad) ─────────────────────── */
static void gaussian_blur(const double *src, double *dst, int H, int W, double sigma) {
    /* Raio do kernel: ceil(3*sigma), mínimo 1 */
    int radius = (int)ceil(3.0 * sigma);
    if (radius < 1) radius = 1;
    int ksize = 2 * radius + 1;

    double *kernel = (double *)malloc(ksize * sizeof(double));
    double sum = 0.0;
    for (int k = 0; k < ksize; k++) {
        double x = k - radius;
        kernel[k] = exp(-0.5 * x * x / (sigma * sigma));
        sum += kernel[k];
    }
    for (int k = 0; k < ksize; k++) kernel[k] /= sum;

    double *tmp = (double *)calloc(H * W, sizeof(double));

    /* Passo horizontal */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double acc = 0.0;
            for (int k = 0; k < ksize; k++) {
                int xi = x + k - radius;
                if (xi >= 0 && xi < W)
                    acc += src[y * W + xi] * kernel[k];
            }
            tmp[y * W + x] = acc;
        }
    }

    /* Passo vertical */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double acc = 0.0;
            for (int k = 0; k < ksize; k++) {
                int yi = y + k - radius;
                if (yi >= 0 && yi < H)
                    acc += tmp[yi * W + x] * kernel[k];
            }
            dst[y * W + x] = acc;
        }
    }

    free(tmp);
    free(kernel);
}

/* ── Clamp double em [0, 1] ─────────────────────────────────────────────── */
static double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

/* ── Gera imagem sintética ─────────────────────────────────────────────────
 * Retorna buffer alocado com H*W valores uint16 (row-major).
 * Caller deve free().
 */
static uint16_t *generate_image(int size) {
    int H = size, W = size, N = H * W;
    RNG rng;
    rng_init(&rng, SEED);

    double *img  = (double *)calloc(N, sizeof(double));
    double *buf  = (double *)malloc(N * sizeof(double));
    double *blur = (double *)malloc(N * sizeof(double));

    /* ── Fundo com textura suave (sigma = 4) ─────────────────────────────── */
    for (int i = 0; i < N; i++) buf[i] = rng_range(&rng, 0.05, 0.25);
    gaussian_blur(buf, blur, H, W, 4.0);
    for (int i = 0; i < N; i++) img[i] += blur[i];

    int r  = size / 6;

    /* ── Região tumoral homogênea — círculo central (sigma = 5) ─────────── */
    int cy = H / 2, cx = W / 2;
    for (int i = 0; i < N; i++) buf[i] = rng_range(&rng, 0.55, 0.90);
    gaussian_blur(buf, blur, H, W, 5.0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int dy = y - cy, dx = x - cx;
            if (dy*dy + dx*dx < r*r)
                img[y*W+x] = blur[y*W+x];
        }
    }

    /* ── Região tumoral heterogênea — deslocada (sigma = 0.4) ───────────── */
    int cy2 = H / 3, cx2 = W / 3, r2 = r / 2;
    for (int i = 0; i < N; i++) buf[i] = rng_range(&rng, 0.0, 1.0);
    gaussian_blur(buf, blur, H, W, 0.4);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int dy = y - cy2, dx = x - cx2;
            if (dy*dy + dx*dx < r2*r2)
                img[y*W+x] = blur[y*W+x];
        }
    }

    /* ── Terceira região — textura média (sigma = 1.5) ───────────────────── */
    int cy3 = H * 2 / 3, cx3 = W * 2 / 3, r3 = r / 3;
    for (int i = 0; i < N; i++) buf[i] = rng_range(&rng, 0.3, 0.7);
    gaussian_blur(buf, blur, H, W, 1.5);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int dy = y - cy3, dx = x - cx3;
            if (dy*dy + dx*dx < r3*r3)
                img[y*W+x] = blur[y*W+x];
        }
    }

    /* ── Clip [0,1] → uint16 [0, 65535] ─────────────────────────────────── */
    uint16_t *out = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t mn = 65535, mx = 0;
    for (int i = 0; i < N; i++) {
        double v = clamp01(img[i]);
        uint16_t u = (uint16_t)(v * 65535.0);
        out[i] = u;
        if (u < mn) mn = u;
        if (u > mx) mx = u;
    }

    printf("  shape=(%d, %d)  dtype=uint16  min=%u  max=%u\n", H, W, mn, mx);

    free(img); free(buf); free(blur);
    return out;
}

/* ── Salva imagem em formato binário ──────────────────────────────────────── */
static int save_image(const char *path, const uint16_t *data, int H, int W) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    uint32_t h = (uint32_t)H, w = (uint32_t)W;
    fwrite(&h, sizeof(uint32_t), 1, f);
    fwrite(&w, sizeof(uint32_t), 1, f);
    fwrite(data, sizeof(uint16_t), (size_t)H * W, f);
    fclose(f);
    return 0;
}

int main(void) {
    /* Criar diretório data/ se não existir */
    mkdir("./data", 0755);

    for (int s = 0; s < N_SIZES; s++) {
        int size = SIZES[s];
        char path[256];
        snprintf(path, sizeof(path), "./data/image_%dx%d.bin", size, size);
        printf("Gerando %s ...\n", path);

        uint16_t *img = generate_image(size);
        if (save_image(path, img, size, size) != 0) {
            free(img);
            return 1;
        }
        free(img);
        printf("  Salvo em: %s\n", path);
    }

    printf("\nTodas as imagens geradas com sucesso.\n");
    return 0;
}
