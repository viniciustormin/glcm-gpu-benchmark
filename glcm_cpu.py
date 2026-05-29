"""
Pipeline CPU sequencial para extração de features GLCM + clustering.
Baseline para comparação com implementações CUDA.
"""

import time
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

N_LEVELS = 64        # níveis de quantização (6 bits)
PATCH_SIZE = 64      # janela de extração
STEP = 32            # passo (overlap 50 %)
DISTANCES = [1]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
N_CLUSTERS = 3
HARALICK_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def quantize(image: np.ndarray) -> np.ndarray:
    """uint16 [0,65535] → uint8 [0,63] via deslocamento de 10 bits."""
    return (image >> 10).astype(np.uint8)


def extract_patches(image_q: np.ndarray):
    H, W = image_q.shape
    patches = []
    for y in range(0, H - PATCH_SIZE + 1, STEP):
        for x in range(0, W - PATCH_SIZE + 1, STEP):
            patches.append(image_q[y : y + PATCH_SIZE, x : x + PATCH_SIZE])
    return patches


def compute_features_cpu(patches: list) -> np.ndarray:
    """Retorna [n_patches, 20] — 5 props × 4 ângulos por patch."""
    features = []
    for patch in patches:
        glcm = graycomatrix(
            patch,
            distances=DISTANCES,
            angles=ANGLES,
            levels=N_LEVELS,
            symmetric=True,
            normed=True,
        )
        feat = np.concatenate(
            [graycoprops(glcm, prop).ravel() for prop in HARALICK_PROPS]
        )
        features.append(feat)
    return np.array(features, dtype=np.float64)


def run_cpu_pipeline(image_path: str) -> dict:
    image = np.load(image_path)

    t0 = time.perf_counter()

    image_q = quantize(image)
    patches = extract_patches(image_q)

    t1 = time.perf_counter()

    features = compute_features_cpu(patches)

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


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./data/image_512x512.npy"
    r = run_cpu_pipeline(path)
    print(f"CPU pipeline — {path}")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
