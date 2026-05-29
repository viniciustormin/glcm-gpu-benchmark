"""
Wrapper ctypes para os kernels CUDA de GLCM + extração de features Haralick em numpy.

Expõe:
  compute_glcm_gpu(patches_uint8, so_path, func_name)
      → np.ndarray [n_patches, 4, N_LEVELS, N_LEVELS]  float32

  glcm_to_haralick_batch(glcm_batch)
      → np.ndarray [n_patches, 20]   (5 props × 4 ângulos)
      Mesmas propriedades que skimage.feature.graycoprops:
        contrast, dissimilarity, homogeneity, energy, correlation

  run_gpu_pipeline(image_path, so_path, func_name) → dict com timings
"""

import ctypes
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from glcm_cpu import N_LEVELS, PATCH_SIZE, STEP, N_CLUSTERS, quantize, extract_patches

# ── Haralick grids (pré-computados, reutilizados para todos os patches) ──────
_I, _J = np.meshgrid(np.arange(N_LEVELS), np.arange(N_LEVELS), indexing="ij")
_I = _I.astype(np.float64)
_J = _J.astype(np.float64)
_DIFF2 = (_I - _J) ** 2
_ABSDIFF = np.abs(_I - _J)
_HOMO_W = 1.0 / (1.0 + _DIFF2)
_IJ = _I * _J
_I2 = _I ** 2
_J2 = _J ** 2


def glcm_to_haralick_batch(glcm_batch: np.ndarray) -> np.ndarray:
    """
    glcm_batch : [n_patches, 4, N_LEVELS, N_LEVELS]  float32 normalizado
    Retorna    : [n_patches, 20]  float64
    """
    G = glcm_batch.astype(np.float64)   # [P, 4, N, N]

    contrast      = np.einsum("paij,ij->pa", G, _DIFF2)
    dissimilarity = np.einsum("paij,ij->pa", G, _ABSDIFF)
    homogeneity   = np.einsum("paij,ij->pa", G, _HOMO_W)
    energy        = np.sqrt(np.einsum("paij,paij->pa", G, G))

    mu_i     = np.einsum("paij,ij->pa", G, _I)      # [P, 4]
    mu_j     = np.einsum("paij,ij->pa", G, _J)
    var_i    = np.einsum("paij,ij->pa", G, _I2) - mu_i ** 2
    var_j    = np.einsum("paij,ij->pa", G, _J2) - mu_j ** 2
    sigma_i  = np.sqrt(np.maximum(var_i, 0.0))
    sigma_j  = np.sqrt(np.maximum(var_j, 0.0))
    cross    = np.einsum("paij,ij->pa", G, _IJ)
    denom    = sigma_i * sigma_j
    correlation = np.where(denom > 1e-10,
                           (cross - mu_i * mu_j) / denom,
                           0.0)

    # Empilhar na mesma ordem que glcm_cpu.py: [prop, angle]
    # → [P, 5, 4] → reshape [P, 20]: [c_a0,c_a1,c_a2,c_a3, d_a0,..., corr_a3]
    props = np.stack([contrast, dissimilarity, homogeneity, energy, correlation],
                     axis=1)          # [P, 5, 4]
    return props.reshape(G.shape[0], -1)   # [P, 20]


# ── Carregamento da biblioteca .so ───────────────────────────────────────────

def _load_lib(so_path: str, func_name: str):
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.restype = None
    func.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # h_patches
        ctypes.c_int,                      # n_patches
        ctypes.c_int,                      # pH
        ctypes.c_int,                      # pW
        ctypes.POINTER(ctypes.c_float),   # h_glcm_out
    ]
    return func


def compute_glcm_gpu(
    patches_uint8: np.ndarray,
    so_path: str,
    func_name: str,
) -> np.ndarray:
    """
    patches_uint8 : [n_patches, pH, pW]  uint8 (valores 0..N_LEVELS-1)
    Retorna       : [n_patches, 4, N_LEVELS, N_LEVELS]  float32
    """
    func = _load_lib(so_path, func_name)

    n, pH, pW = patches_uint8.shape
    patches_c = np.ascontiguousarray(patches_uint8)
    glcm_out = np.zeros(n * 4 * N_LEVELS * N_LEVELS, dtype=np.float32)

    func(
        patches_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n),
        ctypes.c_int(pH),
        ctypes.c_int(pW),
        glcm_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    return glcm_out.reshape(n, 4, N_LEVELS, N_LEVELS)


# ── Pipeline GPU completo ────────────────────────────────────────────────────

def run_gpu_pipeline(image_path: str, so_path: str, func_name: str) -> dict:
    image = np.load(image_path)

    t0 = time.perf_counter()

    image_q = quantize(image)
    patches = extract_patches(image_q)
    patches_arr = np.array(patches, dtype=np.uint8)   # [n, 64, 64]

    t1 = time.perf_counter()

    # GLCM no GPU (inclui H2D, kernel, D2H)
    glcm_batch = compute_glcm_gpu(patches_arr, so_path, func_name)
    # Features Haralick via numpy (CPU, mas rápido em lote)
    features = glcm_to_haralick_batch(glcm_batch)

    t2 = time.perf_counter()

    scaler = StandardScaler()
    features_s = scaler.fit_transform(features)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5)
    labels = km.fit_predict(features_s)
    sil = float(silhouette_score(features_s, labels))

    t3 = time.perf_counter()

    return {
        "total_time_s": t3 - t0,
        "glcm_time_s": t2 - t1,
        "clustering_time_s": t3 - t2,
        "silhouette_score": sil,
        "n_patches": len(patches),
    }
