#!/bin/bash
# ============================================================================
# GLCM Benchmark Final — RTX 4090 (sm_89)
# UFG — Computação de Alto Desempenho
#
# Uso:
#   sbatch job.sh
# ============================================================================

#SBATCH --job-name=glcm_benchmark_final
#SBATCH --partition=rtx4090          # ajustar conforme nome da partição
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=00:45:00
#SBATCH --output=/raid/user_viniciustormin/logs/%x_%j.log
#SBATCH --error=/raid/user_viniciustormin/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
WORKSPACE_ROOT="$(dirname "${PROJECT_ROOT}")"
IMAGE_PATH="/raid/user_viniciustormin/images/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif"
LOG_DIR="/raid/user_viniciustormin/logs"

mkdir -p "${LOG_DIR}"

# ── Detectar runtime ──────────────────────────────────────────────────────────
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

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "GLCM Benchmark Final — $(date)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Runtime : ${RUNTIME}"
echo "Project : ${PROJECT_ROOT}"
echo "============================================================"
echo ""

echo "── GPU Info ────────────────────────────────────────────────"
container_exec nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader
echo ""

# ── Dependências Python ───────────────────────────────────────────────────────
echo "── Instalando dependências Python ──────────────────────────"
container_exec pip install --user --quiet scikit-image scikit-learn
echo "Instalação concluída."
echo ""

# ── Compilar kernels CUDA (sm_89 = RTX 4090) ─────────────────────────────────
echo "── Compilando kernels CUDA (sm_89) ─────────────────────────"

compile_kernel() {
    local src="$1" out="$2" extra="${3:-}"
    echo -n "  ${out} ... "
    container_exec nvcc -O3 -arch=sm_89 -shared -Xcompiler -fPIC \
        ${extra} -o "${PROJECT_ROOT}/${out}" "${PROJECT_ROOT}/${src}"
    echo "OK"
}

compile_kernel glcm_cuda.cu      glcm_cuda.so
compile_kernel glcm_shared.cu    glcm_shared.so
compile_kernel glcm_shared_v2.cu glcm_shared_v2.so

# Dynamic Parallelism requer -rdc=true e linkagem extra
echo -n "  glcm_dynpar.so ... "
container_exec nvcc -O3 -arch=sm_89 -rdc=true -shared -Xcompiler -fPIC \
    -o "${PROJECT_ROOT}/glcm_dynpar.so" \
       "${PROJECT_ROOT}/glcm_dynpar.cu" \
    -lcudadevrt -lcudart_static -lrt -lpthread -ldl
echo "OK"
echo ""

# ── Gerar dados sintéticos ────────────────────────────────────────────────────
echo "── Gerando imagens sintéticas ──────────────────────────────"
container_exec python glcm_data.py
echo ""

# ── Benchmark ─────────────────────────────────────────────────────────────────
echo "── Executando benchmark final ──────────────────────────────"
container_exec python benchmark.py
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
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
    hdr = f"{'Approach':<30} {'Resolution':<12} {'Total(s)':<10} {'GLCM(s)':<10} {'Speedup':<10} Silhouette"
    print(hdr)
    print("-" * len(hdr))
    for r in d["runs"]:
        print(
            f"{r['approach']:<30} {r['resolution']:<12} "
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
