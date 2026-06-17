"""
Gerador de imagens sintéticas uint16 com texturas variadas.
Simula regiões tumorais homogêneas e heterogêneas (inspirado no paper CHASM).
Salva em ./data/image_{size}x{size}.npy
"""

import numpy as np
from skimage.filters import gaussian
import os

SIZES = [256, 512, 1024]


def generate_image(size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)

    # Fundo com textura suave (tecido normal)
    bg = rng.uniform(0.05, 0.25, (size, size))
    img += gaussian(bg, sigma=4)

    y, x = np.ogrid[:size, :size]
    r = size // 6

    # Região tumoral homogênea (círculo central) — alta intensidade, sigma grande
    cy, cx = size // 2, size // 2
    mask_homog = (y - cy) ** 2 + (x - cx) ** 2 < r ** 2
    homog = rng.uniform(0.55, 0.90, (size, size))
    img[mask_homog] = gaussian(homog, sigma=5)[mask_homog]

    # Região tumoral heterogênea (offset) — intensidade variada, sigma pequeno
    cy2, cx2 = size // 3, size // 3
    r2 = r // 2
    mask_heterog = (y - cy2) ** 2 + (x - cx2) ** 2 < r2 ** 2
    heterog = rng.uniform(0.0, 1.0, (size, size))
    img[mask_heterog] = gaussian(heterog, sigma=0.4)[mask_heterog]

    # Terceira região: textura média (outro quadrante)
    cy3, cx3 = size * 2 // 3, size * 2 // 3
    r3 = r // 3
    mask_mid = (y - cy3) ** 2 + (x - cx3) ** 2 < r3 ** 2
    mid = rng.uniform(0.3, 0.7, (size, size))
    img[mask_mid] = gaussian(mid, sigma=1.5)[mask_mid]

    img = np.clip(img, 0.0, 1.0)
    return (img * 65535).astype(np.uint16)


def main():
    os.makedirs("./data", exist_ok=True)
    for size in SIZES:
        arr = generate_image(size)
        path = f"./data/image_{size}x{size}.npy"
        np.save(path, arr)
        print(f"Saved {path}  shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min()}  max={arr.max()}")


if __name__ == "__main__":
    main()
