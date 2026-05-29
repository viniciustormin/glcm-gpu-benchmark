#!/bin/bash
# ============================================================================
# GLCM Benchmark — H100 (sm_90)
# UFG — Computação de Alto Desempenho
#
# Uso:
#   sbatch job.sh
#
# Requer no cluster:
#   /raid/user_viniciustormin/images/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif
# ============================================================================

#SBATCH --job-name=glcm_benchmark
#SBATCH --partition=h100n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_viniciustormin/logs/%x_%j.log
#SBATCH --error=/raid/user_viniciustormin/logs/%x_%j.err

set -euo pipefail

# ── Caminhos ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
WORKSPACE_ROOT="$(dirname "${PROJECT_ROOT}")"
IMAGE_PATH="/raid/user_viniciustormin/images/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif"
LOG_DIR="/raid/user_viniciustormin/logs"

mkdir -p "${LOG_DIR}"

# ── Detectar runtime de container ────────────────────────────────────────────
if command -v apptainer &>/dev/null; then
    RUNTIME="apptainer"
elif command -v singularity &>/dev/null; then
    RUNTIME="singularity"
else
    echo "ERRO: nem apptainer nem singularity encontrado no PATH." >&2
    exit 1
fi

CONTAINER_ARGS=(
    "${RUNTIME}" exec --nv
    --bind "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}"
    --pwd  "${PROJECT_ROOT}"
    "${IMAGE_PATH}"
)

container_exec() { "${CONTAINER_ARGS[@]}" "$@"; }

# ── Header do log ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "GLCM Benchmark — $(date)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Runtime : ${RUNTIME}"
echo "Image   : ${IMAGE_PATH}"
echo "Project : ${PROJECT_ROOT}"
echo "============================================================"
echo ""

echo "── GPU Info ────────────────────────────────────────────────"
container_exec nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader
echo ""

# ── Dependências Python ───────────────────────────────────────────────────────
# O container pytorch:2.8.0-cuda12.8-cudnn9-devel já inclui:
#   numpy, scipy, torch — precisamos apenas de scikit-image e scikit-learn.
echo "── Instalando dependências Python ──────────────────────────"
container_exec pip install --user --quiet scikit-image scikit-learn
echo "Instalação concluída."
echo ""

# ── Compilar kernels CUDA ────────────────────────────────────────────────────
echo "── Compilando kernels CUDA (sm_90) ─────────────────────────"

echo -n "  glcm_cuda.so  ... "
container_exec nvcc -O3 -arch=sm_90 -shared -fPIC \
    -o "${PROJECT_ROOT}/glcm_cuda.so" \
       "${PROJECT_ROOT}/glcm_cuda.cu"
echo "OK"

echo -n "  glcm_shared.so ... "
container_exec nvcc -O3 -arch=sm_90 -shared -fPIC \
    -o "${PROJECT_ROOT}/glcm_shared.so" \
       "${PROJECT_ROOT}/glcm_shared.cu"
echo "OK"
echo ""

# ── Gerar dados sintéticos ────────────────────────────────────────────────────
echo "── Gerando imagens sintéticas ──────────────────────────────"
container_exec python glcm_data.py
echo ""

# ── Benchmark ─────────────────────────────────────────────────────────────────
echo "── Executando benchmark ────────────────────────────────────"
container_exec python benchmark.py
echo ""

# ── Summary final ─────────────────────────────────────────────────────────────
echo "── Resultados finais ───────────────────────────────────────"
container_exec python - <<'PYEOF'
import json, sys
try:
    with open("benchmark_results.json") as f:
        d = json.load(f)
    print(f"GPU  : {d['gpu']}")
    print(f"Arch : {d['cuda_arch']}")
    print(f"cuML : {d['cuml_available']}")
    print()
    hdr = f"{'Approach':<28} {'Resolution':<12} {'Total(s)':<10} {'GLCM(s)':<10} {'Speedup':<10} Silhouette"
    print(hdr)
    print("-" * len(hdr))
    for r in d["runs"]:
        print(
            f"{r['approach']:<28} {r['resolution']:<12} "
            f"{r['total_time_s']:<10.3f} {r['glcm_time_s']:<10.3f} "
            f"{r['speedup']:<10.2f}x {r['silhouette_score']:.3f}"
        )
except Exception as e:
    print(f"[ERRO ao ler benchmark_results.json] {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "============================================================"
echo "FIM — $(date)"
echo "============================================================"
