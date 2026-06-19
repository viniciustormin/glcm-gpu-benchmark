/*
 * benchmark.c  —  Benchmark master: CPU vs todos os kernels CUDA.
 * Salva benchmark_results.json.
 *
 * Compile (junto com glcm_cpu.c e glcm_pipeline.c):
 *   gcc -O3 -std=c11 -Wno-unused-result -o benchmark \
 *       benchmark.c glcm_cpu.c glcm_pipeline.c -lm -ldl
 */

#define _POSIX_C_SOURCE 200809L
#include "glcm_common.h"
#include <dlfcn.h>

/* Declarações das funções dos outros módulos */
RunResult run_cpu_pipeline(const char *image_path);
RunResult run_gpu_pipeline(const char *image_path,
                            const char *so_path,
                            const char *func_name);
void close_gpu_lib(void);

/* ── Configuração ─────────────────────────────────────────────────────────── */
#define N_REPS       3
#define N_SIZES      3
#define N_APPROACHES 5

static const int SIZES[N_SIZES] = {256, 512, 1024};

typedef enum { KIND_CPU, KIND_GPU } Kind;
typedef struct {
    const char *name;
    Kind        kind;
    const char *so_path;
    const char *func_name;
} Approach;

static const Approach APPROACHES[N_APPROACHES] = {
    { "CPU Sequential",        KIND_CPU, NULL,                  NULL                     },
    { "CUDA Basic",            KIND_GPU, "./glcm_cuda.so",      "compute_glcm_basic"     },
    { "CUDA Shared Memory",    KIND_GPU, "./glcm_shared.so",    "compute_glcm_shared"    },
    { "CUDA Shared v2 (fix)",  KIND_GPU, "./glcm_shared_v2.so", "compute_glcm_shared_v2" },
    { "CUDA Dyn. Parallelism", KIND_GPU, "./glcm_dynpar.so",    "compute_glcm_dynpar"    },
};

/* ── Utilitários ──────────────────────────────────────────────────────────── */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static void get_gpu_name(char *buf, int len) {
    FILE *f = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (!f || !fgets(buf, len, f)) { snprintf(buf, len, "unknown"); }
    else { int l = strlen(buf); while (l > 0 && (buf[l-1]=='\n'||buf[l-1]=='\r')) buf[--l]=0; }
    if (f) pclose(f);
}

static void get_cuda_arch(char *buf, int len) {
    FILE *f = popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null", "r");
    char tmp[32] = {0};
    if (f && fgets(tmp, sizeof(tmp), f)) {
        int maj=0, min=0;
        if (sscanf(tmp, "%d.%d", &maj, &min) == 2)
            snprintf(buf, len, "sm_%d%d", maj, min);
        else snprintf(buf, len, "unknown");
    } else snprintf(buf, len, "unknown");
    if (f) pclose(f);
}

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) { if (*s=='"'||*s=='\\') fputc('\\',f); fputc(*s,f); }
    fputc('"', f);
}

/* ── Resultado por linha ──────────────────────────────────────────────────── */
typedef struct {
    char   approach[64], resolution[16];
    double total_time_s, glcm_time_s, clustering_time_s, silhouette_score, speedup;
} BenchRow;

/* ── Mediana de N_REPS execuções ─────────────────────────────────────────── */
static RunResult median_run(const char *img, const Approach *ap) {
    RunResult results[N_REPS];
    double    times[N_REPS];
    for (int r = 0; r < N_REPS; r++) {
        results[r] = (ap->kind == KIND_CPU)
            ? run_cpu_pipeline(img)
            : run_gpu_pipeline(img, ap->so_path, ap->func_name);
        times[r] = results[r].total_time_s;
    }
    double sorted[N_REPS];
    memcpy(sorted, times, sizeof(times));
    qsort(sorted, N_REPS, sizeof(double), cmp_double);
    double med = sorted[N_REPS / 2];
    for (int r = 0; r < N_REPS; r++)
        if (fabs(results[r].total_time_s - med) < 1e-9) return results[r];
    return results[0];
}

/* ── Warmup GPU ──────────────────────────────────────────────────────────── */
static void warmup_gpu(const Approach *ap) {
    const char *path = "./data/image_256x256.bin";
    if (ap->kind == KIND_GPU && file_exists(path) && file_exists(ap->so_path))
        run_gpu_pipeline(path, ap->so_path, ap->func_name);
}

/* ═════════════════════════════════════════════════════════════════════════════*/
int main(void) {
    char gpu_name[256], cuda_arch[32];
    get_gpu_name(gpu_name, sizeof(gpu_name));
    get_cuda_arch(cuda_arch, sizeof(cuda_arch));

    printf("=================================================================\n");
    printf("GLCM Benchmark — CPU vs CUDA Basic vs Shared vs Shared v2 vs DynPar\n");
    printf("=================================================================\n");
    printf("GPU       : %s\n", gpu_name);
    printf("CUDA arch : %s\n\n", cuda_arch);

    /* Verificar imagens */
    for (int s = 0; s < N_SIZES; s++) {
        char p[64]; snprintf(p, sizeof(p), "./data/image_%dx%d.bin", SIZES[s], SIZES[s]);
        if (!file_exists(p)) {
            fprintf(stderr, "[ERRO] %s não encontrado. Execute: ./glcm_data\n", p);
            return 1;
        }
    }

    /* Avisar .so ausentes */
    for (int a = 1; a < N_APPROACHES; a++)
        if (!file_exists(APPROACHES[a].so_path))
            fprintf(stderr, "[AVISO] .so ausente, será pulado: %s\n", APPROACHES[a].so_path);

    /* Warmup — uma chamada descartada por kernel para inicializar contexto CUDA */
    printf("Aquecendo GPU...\n");
    for (int a = 1; a < N_APPROACHES; a++) warmup_gpu(&APPROACHES[a]);
    printf("Warmup concluído.\n\n");

    BenchRow rows[N_SIZES * N_APPROACHES];
    int n_rows = 0;
    double cpu_ref[N_SIZES];

    for (int s = 0; s < N_SIZES; s++) {
        char img[64], res_label[16];
        snprintf(img,       sizeof(img),       "./data/image_%dx%d.bin", SIZES[s], SIZES[s]);
        snprintf(res_label, sizeof(res_label),  "%dx%d", SIZES[s], SIZES[s]);

        printf("── Resolução %s ─────────────────────────────\n", res_label);
        cpu_ref[s] = -1.0;

        for (int a = 0; a < N_APPROACHES; a++) {
            const Approach *ap = &APPROACHES[a];
            if (ap->kind == KIND_GPU && !file_exists(ap->so_path)) {
                printf("  %-30s [PULADO]\n", ap->name); continue;
            }

            RunResult r = median_run(img, ap);
            double speedup = 1.0;
            if (ap->kind == KIND_CPU) cpu_ref[s] = r.total_time_s;
            else if (cpu_ref[s] > 0)  speedup = cpu_ref[s] / r.total_time_s;

            printf("  %-30s total=%.3fs  glcm=%.3fs  speedup=%.2fx  sil=%.3f\n",
                   ap->name, r.total_time_s, r.glcm_time_s, speedup, r.silhouette_score);

            BenchRow *row = &rows[n_rows++];
            snprintf(row->approach,   sizeof(row->approach),   "%s", ap->name);
            snprintf(row->resolution, sizeof(row->resolution), "%s", res_label);
            row->total_time_s      = r.total_time_s;
            row->glcm_time_s       = r.glcm_time_s;
            row->clustering_time_s = r.clustering_time_s;
            row->silhouette_score  = r.silhouette_score;
            row->speedup           = speedup;
        }
        printf("\n");
    }

    close_gpu_lib();

    /* JSON */
    FILE *jf = fopen("benchmark_results.json", "w");
    if (jf) {
        fprintf(jf, "{\n  \"gpu\": "); json_str(jf, gpu_name);
        fprintf(jf, ",\n  \"cuda_arch\": "); json_str(jf, cuda_arch);
        fprintf(jf, ",\n  \"cuml_available\": false,\n  \"runs\": [\n");
        for (int i = 0; i < n_rows; i++) {
            const BenchRow *r = &rows[i];
            fprintf(jf, "    {\n");
            fprintf(jf, "      \"approach\": ");   json_str(jf, r->approach);   fprintf(jf, ",\n");
            fprintf(jf, "      \"resolution\": "); json_str(jf, r->resolution); fprintf(jf, ",\n");
            fprintf(jf, "      \"total_time_s\": %.6f,\n",      r->total_time_s);
            fprintf(jf, "      \"glcm_time_s\": %.6f,\n",       r->glcm_time_s);
            fprintf(jf, "      \"clustering_time_s\": %.6f,\n", r->clustering_time_s);
            fprintf(jf, "      \"silhouette_score\": %.6f,\n",  r->silhouette_score);
            fprintf(jf, "      \"speedup\": %.4f\n",             r->speedup);
            fprintf(jf, "    }%s\n", i < n_rows-1 ? "," : "");
        }
        fprintf(jf, "  ]\n}\n");
        fclose(jf);
        printf("Resultados salvos em: benchmark_results.json\n\n");
    }

    /* Tabela resumida */
    printf("%-30s %-12s %-10s %-10s %-10s %s\n",
           "Approach","Resolution","Total(s)","GLCM(s)","Speedup","Silhouette");
    printf("%.*s\n", 82, "--------------------------------------------------------------------------------"
           "--------------------------------------------------------------------------------");
    for (int i = 0; i < n_rows; i++) {
        const BenchRow *r = &rows[i];
        printf("%-30s %-12s %-10.3f %-10.3f %-9.2fx %.3f\n",
               r->approach, r->resolution,
               r->total_time_s, r->glcm_time_s,
               r->speedup, r->silhouette_score);
    }
    return 0;
}
