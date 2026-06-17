"""
Benchmark final: CPU Sequential vs CUDA Basic vs CUDA Shared Memory
vs CUDA Shared v2 (merge condicional) vs CUDA Dynamic Parallelism
vs CUDA Basic + cuML KMeans.

Resoluções: 256x256, 512x512, 1024x1024.
3 repetições por combinação → usa a mediana.
Salva benchmark_results.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from glcm_cpu import run_cpu_pipeline
from glcm_gpu import run_gpu_pipeline, run_gpu_pipeline_cuml

RESOLUTIONS = [256, 512, 1024]
N_REPS = 3
DATA_DIR = Path("./data")

# (nome, tipo, so_path, func_name, usa_cuml)
APPROACHES = [
    ("CPU Sequential",          "cpu",    None,                    None,                       False),
    ("CUDA Basic",              "gpu",    "./glcm_cuda.so",        "compute_glcm_basic",        False),
    ("CUDA Shared Memory",      "gpu",    "./glcm_shared.so",      "compute_glcm_shared",       False),
    ("CUDA Shared v2 (fix)",    "gpu",    "./glcm_shared_v2.so",   "compute_glcm_shared_v2",    False),
    ("CUDA Dyn. Parallelism",   "gpu",    "./glcm_dynpar.so",      "compute_glcm_dynpar",       False),
    ("CUDA Basic + cuML",       "gpu",    "./glcm_cuda.so",        "compute_glcm_basic",        True),
]


def get_gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
    except Exception:
        return "unknown"


def check_cuml() -> bool:
    try:
        import cuml  # noqa
        return True
    except ImportError:
        return False


def image_path(size: int) -> str:
    return str(DATA_DIR / f"image_{size}x{size}.npy")


def median_run(fn, n_reps: int) -> dict:
    results = [fn() for _ in range(n_reps)]
    results.sort(key=lambda r: r["total_time_s"])
    return results[n_reps // 2]


def warmup_gpu(so_path: str, func_name: str, use_cuml: bool):
    path = image_path(256)
    if Path(path).exists() and Path(so_path).exists():
        try:
            if use_cuml:
                run_gpu_pipeline_cuml(path, so_path, func_name)
            else:
                run_gpu_pipeline(path, so_path, func_name)
        except Exception:
            pass


def main():
    print("=" * 75)
    print("GLCM Benchmark — CPU vs CUDA Basic vs Shared vs Shared v2 vs DynPar vs cuML")
    print("=" * 75)

    gpu_name    = get_gpu_name()
    cuml_avail  = check_cuml()
    # Detectar arch da GPU
    try:
        arch_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0].replace(".", "")
        cuda_arch = f"sm_{arch_out}"
    except Exception:
        cuda_arch = "sm_89"

    print(f"GPU            : {gpu_name}")
    print(f"CUDA arch      : {cuda_arch}")
    print(f"cuML available : {cuml_avail}")
    print()

    for size in RESOLUTIONS:
        p = image_path(size)
        if not Path(p).exists():
            print(f"[ERROR] Arquivo não encontrado: {p}")
            print("Execute primeiro: python glcm_data.py")
            sys.exit(1)

    # Verificar .so
    for name, kind, so, func, use_cuml in APPROACHES:
        if kind == "gpu" and so and not Path(so).exists():
            print(f"[AVISO] .so não encontrado, pulando: {so}  ({name})")

    # Warmup
    print("Aquecendo GPU...")
    for _, kind, so, func, use_cuml in APPROACHES:
        if kind == "gpu" and so and Path(so).exists():
            warmup_gpu(so, func, use_cuml)
    print("Warmup concluído.\n")

    cpu_ref: dict[int, float] = {}
    runs = []

    for size in RESOLUTIONS:
        path = image_path(size)
        res_label = f"{size}x{size}"
        print(f"── Resolução {res_label} ─────────────────────────────────────")

        for approach_name, kind, so, func, use_cuml in APPROACHES:
            # Pular se .so ausente
            if kind == "gpu" and so and not Path(so).exists():
                print(f"  {approach_name:<30} [PULADO — .so ausente]")
                continue
            if use_cuml and not cuml_avail:
                print(f"  {approach_name:<30} [PULADO — cuML não disponível]")
                continue

            if kind == "cpu":
                fn = lambda p=path: run_cpu_pipeline(p)
            elif use_cuml:
                fn = lambda p=path, s=so, f=func: run_gpu_pipeline_cuml(p, s, f)
            else:
                fn = lambda p=path, s=so, f=func: run_gpu_pipeline(p, s, f)

            print(f"  {approach_name:<30} ", end="", flush=True)
            result = median_run(fn, N_REPS)

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
        "cuda_arch": cuda_arch,
        "cuml_available": cuml_avail,
        "runs": runs,
    }

    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Resultados salvos em: {out_path}\n")

    print(f"{'Approach':<30} {'Resolution':<12} {'Total(s)':<10} {'GLCM(s)':<10} {'Speedup':<10} Silhouette")
    print("-" * 80)
    for r in runs:
        print(
            f"{r['approach']:<30} {r['resolution']:<12} "
            f"{r['total_time_s']:<10.3f} {r['glcm_time_s']:<10.3f} "
            f"{r['speedup']:<10.2f}x {r['silhouette_score']:.3f}"
        )


if __name__ == "__main__":
    main()
