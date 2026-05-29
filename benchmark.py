"""
Benchmark: CPU Sequential vs CUDA Basic vs CUDA + Shared Memory.
Resoluções: 256x256, 512x512, 1024x1024.
3 repetições por combinação → usa a mediana.
Salva benchmark_results.json no formato exato exigido.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from glcm_cpu import run_cpu_pipeline
from glcm_gpu import run_gpu_pipeline

# ── Configuração ──────────────────────────────────────────────────────────────

RESOLUTIONS = [256, 512, 1024]
N_REPS = 3
DATA_DIR = Path("./data")
SO_BASIC  = "./glcm_cuda.so"
SO_SHARED = "./glcm_shared.so"

APPROACHES = [
    ("CPU Sequential",      "cpu",    None,       None),
    ("CUDA Basic",          "gpu",    SO_BASIC,   "compute_glcm_basic"),
    ("CUDA Shared Memory",  "gpu",    SO_SHARED,  "compute_glcm_shared"),
]


# ── Utilitários ───────────────────────────────────────────────────────────────

def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        return out
    except Exception:
        return "unknown"


def check_cuml() -> bool:
    try:
        import cuml  # noqa: F401
        return True
    except ImportError:
        return False


def image_path(size: int) -> str:
    return str(DATA_DIR / f"image_{size}x{size}.npy")


def median_run(fn, n_reps: int) -> dict:
    """Executa fn() n_reps vezes e retorna o resultado com tempo mediano."""
    results = [fn() for _ in range(n_reps)]
    # Ordena pelo tempo total e pega o mediano
    results.sort(key=lambda r: r["total_time_s"])
    return results[n_reps // 2]


def warmup_gpu(so_path: str, func_name: str):
    """Aquece a GPU com uma chamada descartada na menor imagem."""
    path = image_path(256)
    if Path(path).exists():
        try:
            run_gpu_pipeline(path, so_path, func_name)
        except Exception:
            pass


# ── Benchmark principal ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GLCM Benchmark — CPU vs CUDA Basic vs CUDA Shared Memory")
    print("=" * 70)

    gpu_name = get_gpu_name()
    cuml_available = check_cuml()
    print(f"GPU            : {gpu_name}")
    print(f"CUDA arch      : sm_90")
    print(f"cuML available : {cuml_available}")
    print()

    # Verificar que os dados existem
    for size in RESOLUTIONS:
        p = image_path(size)
        if not Path(p).exists():
            print(f"[ERROR] Arquivo não encontrado: {p}")
            print("Execute primeiro: python glcm_data.py")
            sys.exit(1)

    # Verificar que os .so existem
    for _, kind, so, _ in APPROACHES:
        if kind == "gpu" and not Path(so).exists():
            print(f"[ERROR] Biblioteca não encontrada: {so}")
            print("Compile os kernels CUDA antes de rodar o benchmark.")
            sys.exit(1)

    # Warmup GPU
    print("Aquecendo GPU...")
    for _, kind, so, func in APPROACHES:
        if kind == "gpu":
            warmup_gpu(so, func)
    print("Warmup concluído.\n")

    # Referência CPU por resolução (para calcular speedup)
    cpu_ref: dict[int, float] = {}

    runs = []

    for size in RESOLUTIONS:
        path = image_path(size)
        res_label = f"{size}x{size}"
        print(f"── Resolução {res_label} ──────────────────────────────")

        for approach_name, kind, so, func in APPROACHES:
            if kind == "cpu":
                fn = lambda p=path: run_cpu_pipeline(p)
            else:
                fn = lambda p=path, s=so, f=func: run_gpu_pipeline(p, s, f)

            print(f"  {approach_name:<28} ", end="", flush=True)
            t_wall = time.perf_counter()
            result = median_run(fn, N_REPS)
            t_wall = time.perf_counter() - t_wall

            if kind == "cpu":
                cpu_ref[size] = result["total_time_s"]
                speedup = 1.0
            else:
                ref = cpu_ref.get(size, result["total_time_s"])
                speedup = ref / result["total_time_s"] if result["total_time_s"] > 0 else 0.0

            print(
                f"total={result['total_time_s']:.3f}s  "
                f"glcm={result['glcm_time_s']:.3f}s  "
                f"speedup={speedup:.2f}x  "
                f"sil={result['silhouette_score']:.3f}"
            )

            runs.append({
                "approach": approach_name,
                "resolution": res_label,
                "total_time_s": round(result["total_time_s"], 6),
                "glcm_time_s": round(result["glcm_time_s"], 6),
                "clustering_time_s": round(result["clustering_time_s"], 6),
                "silhouette_score": round(result["silhouette_score"], 6),
                "speedup": round(speedup, 4),
            })

        print()

    output = {
        "gpu": gpu_name,
        "cuda_arch": "sm_90",
        "cuml_available": cuml_available,
        "runs": runs,
    }

    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Resultados salvos em: {out_path}")
    print()

    # Tabela final resumida
    print(f"{'Approach':<28} {'Resolution':<12} {'Total(s)':<10} {'Speedup':<10} {'Silhouette'}")
    print("-" * 75)
    for r in runs:
        print(
            f"{r['approach']:<28} {r['resolution']:<12} "
            f"{r['total_time_s']:<10.3f} {r['speedup']:<10.2f}x "
            f"{r['silhouette_score']:.3f}"
        )


if __name__ == "__main__":
    main()
