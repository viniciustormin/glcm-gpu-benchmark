"""
glcm_gpu.py — Interface Python/ctypes para os kernels CUDA de GLCM.
Compatível com: glcm_cuda.so, glcm_shared.so, glcm_shared_v2.so, glcm_dynpar.so
"""

import ctypes
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

N_LEVELS = 64
PATCH_SIZE = 64
STEP = 32
DISTANCES = [1]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
N_CLUSTERS = 3
HARALICK_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def quantize(image: np.ndarray) -> np.ndarray:
    return (image >> 10).astype(np.uint8)


def extract_patches(image_q: np.ndarray):
    H, W = image_q.shape
    patches = []
    for y in range(0, H - PATCH_SIZE + 1, STEP):
        for x in range(0, W - PATCH_SIZE + 1, STEP):
            patches.append(image_q[y: y + PATCH_SIZE, x: x + PATCH_SIZE])
    return patches


def compute_haralick_from_glcm(glcm_batch: np.ndarray) -> np.ndarray:
    """
    Calcula 5 propriedades de Haralick a partir de GLCMs pré-calculadas.
    glcm_batch: [n_patches, 4, N_LEVELS, N_LEVELS] (float32, normalizada)
    Retorna: [n_patches, 20]  (5 props × 4 ângulos)
    """
    n_patches, n_angles, L, _ = glcm_batch.shape
    features = np.zeros((n_patches, n_angles * 5), dtype=np.float64)

    i_idx = np.arange(L, dtype=np.float64)
    j_idx = np.arange(L, dtype=np.float64)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')

    for p in range(n_patches):
        for a in range(n_angles):
            g = glcm_batch[p, a].astype(np.float64)
            # Contrast
            contrast = np.sum((I - J) ** 2 * g)
            # Dissimilarity
            dissim = np.sum(np.abs(I - J) * g)
            # Homogeneity
            homog = np.sum(g / (1.0 + (I - J) ** 2))
            # Energy
            energy = np.sum(g ** 2)
            # Correlation
            mu_i = np.sum(I * g)
            mu_j = np.sum(J * g)
            sig_i = np.sqrt(np.sum((I - mu_i) ** 2 * g) + 1e-10)
            sig_j = np.sqrt(np.sum((J - mu_j) ** 2 * g) + 1e-10)
            corr = np.sum((I - mu_i) * (J - mu_j) * g) / (sig_i * sig_j)
            offset = a * 5
            features[p, offset + 0] = contrast
            features[p, offset + 1] = dissim
            features[p, offset + 2] = homog
            features[p, offset + 3] = energy
            features[p, offset + 4] = corr

    return features


def run_gpu_pipeline(image_path: str, so_path: str, func_name: str) -> dict:
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, func_name)
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    image = np.load(image_path)

    t0 = time.perf_counter()

    image_q = quantize(image)
    patches = extract_patches(image_q)
    n_patches = len(patches)

    patch_arr = np.stack(patches, axis=0).astype(np.uint8)  # [n, H, W]
    patch_flat = np.ascontiguousarray(patch_arr)
    pH, pW = PATCH_SIZE, PATCH_SIZE

    glcm_out = np.zeros((n_patches, 4, N_LEVELS, N_LEVELS), dtype=np.float32)
    glcm_flat = np.ascontiguousarray(glcm_out)

    t1 = time.perf_counter()

    fn(
        patch_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n_patches),
        ctypes.c_int(pH),
        ctypes.c_int(pW),
        glcm_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    glcm_out = glcm_flat.reshape(n_patches, 4, N_LEVELS, N_LEVELS)
    features = compute_haralick_from_glcm(glcm_out)

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
        "n_patches": n_patches,
    }


def run_gpu_pipeline_cuml(image_path: str, so_path: str, func_name: str) -> dict:
    """
    Variante com cuML KMeans (GPU) em vez do sklearn (CPU).
    Requer: pip install cuml-cu12
    """
    try:
        from cuml.cluster import KMeans as cuKMeans
        from cuml.metrics import silhouette_score as cu_sil
        import cupy as cp
        USE_CUML = True
    except ImportError:
        USE_CUML = False

    if not USE_CUML:
        return run_gpu_pipeline(image_path, so_path, func_name)

    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, func_name)
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    image = np.load(image_path)
    t0 = time.perf_counter()

    image_q = quantize(image)
    patches = extract_patches(image_q)
    n_patches = len(patches)
    patch_arr = np.ascontiguousarray(np.stack(patches, axis=0).astype(np.uint8))
    pH, pW = PATCH_SIZE, PATCH_SIZE

    glcm_flat = np.zeros((n_patches, 4, N_LEVELS, N_LEVELS), dtype=np.float32)
    glcm_flat = np.ascontiguousarray(glcm_flat)

    t1 = time.perf_counter()

    fn(
        patch_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n_patches), ctypes.c_int(pH), ctypes.c_int(pW),
        glcm_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    glcm_out = glcm_flat.reshape(n_patches, 4, N_LEVELS, N_LEVELS)
    features = compute_haralick_from_glcm(glcm_out)

    t2 = time.perf_counter()

    # cuML KMeans (GPU)
    scaler = StandardScaler()
    features_s = scaler.fit_transform(features).astype(np.float32)
    features_gpu = cp.asarray(features_s)
    km = cuKMeans(n_clusters=N_CLUSTERS, random_state=42, max_iter=300)
    labels_gpu = km.fit_predict(features_gpu)
    sil = float(cu_sil(features_gpu, labels_gpu))

    t3 = time.perf_counter()

    return {
        "total_time_s": t3 - t0,
        "glcm_time_s": t2 - t1,
        "clustering_time_s": t3 - t2,
        "silhouette_score": sil,
        "n_patches": n_patches,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./data/image_512x512.npy"
    so   = sys.argv[2] if len(sys.argv) > 2 else "./glcm_cuda.so"
    fn   = sys.argv[3] if len(sys.argv) > 3 else "compute_glcm_basic"
    r = run_gpu_pipeline(path, so, fn)
    print(f"GPU pipeline [{fn}] — {path}")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
