"""
utils/spectral.py — Spectral feature extraction and index building for NAIP RGBI chips.

NAIP band order: 1=Red, 2=Green, 3=Blue, 4=NIR (near-infrared)

Indices computed per chip
─────────────────────────
  NDVI   (NIR - R) / (NIR + R)              vegetation density
  NDWI   (G - NIR) / (G + NIR)              open water
  EVI    2.5 * (NIR - R) / (NIR + 6R - 7.5B + 1)   enhanced vegetation
  SAVI   ((NIR - R) / (NIR + R + 0.5)) * 1.5        soil-adjusted veg
  NDBI   (SWIR - NIR) / (SWIR + NIR)        built-up (approx via R as SWIR proxy)
  Brightness  mean of all 4 bands           overall albedo

Each chip produces a 12-dimensional spectral feature vector:
  [ndvi_mean, ndvi_std, ndwi_mean, ndwi_std,
   evi_mean,  evi_std,  savi_mean, savi_std,
   ndbi_mean, ndbi_std, bright_mean, bright_std]

This vector is L2-normalized and indexed with FAISS for cosine search,
just like the visual embeddings.

Query by description
────────────────────
build_spectral_query_vector() maps natural-language descriptions to a
synthetic spectral vector so users can query with phrases like
"dense vegetation", "open water", "impervious surface", "bare soil".
"""

from pathlib import Path
import numpy as np
import faiss

import sys
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import CACHE_DIR


# ── Spectral index catalogue ──────────────────────────────────────────────────
# Index names in the order they appear in the feature vector
SPECTRAL_INDEX_NAMES = [
    "ndvi_mean", "ndvi_std",
    "ndwi_mean", "ndwi_std",
    "evi_mean",  "evi_std",
    "savi_mean", "savi_std",
    "ndbi_mean", "ndbi_std",
    "brightness_mean", "brightness_std",
]
SPECTRAL_DIM = len(SPECTRAL_INDEX_NAMES)  # 12


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with zero-division guard."""
    denom = a + b
    return np.where(np.abs(denom) > 1e-6, (a - b) / denom, 0.0)


def compute_chip_spectral(chip4: np.ndarray) -> np.ndarray:
    """
    Compute spectral feature vector for a single 4-band chip.

    Parameters
    ----------
    chip4 : np.ndarray  shape (4, H, W)  float32 in [0, 1]
              bands: [R, G, B, NIR]

    Returns
    -------
    vec : np.ndarray  shape (SPECTRAL_DIM,)  float32
    """
    R, G, B, NIR = chip4[0], chip4[1], chip4[2], chip4[3]

    # ── NDVI ──────────────────────────────────────────────────────────────────
    ndvi = _safe_ratio(NIR, R)

    # ── NDWI ──────────────────────────────────────────────────────────────────
    ndwi = _safe_ratio(G, NIR)

    # ── EVI ───────────────────────────────────────────────────────────────────
    evi_num   = NIR - R
    evi_denom = NIR + 6.0 * R - 7.5 * B + 1.0
    evi = np.where(np.abs(evi_denom) > 1e-6, 2.5 * evi_num / evi_denom, 0.0)
    evi = np.clip(evi, -2.0, 2.0)

    # ── SAVI ──────────────────────────────────────────────────────────────────
    L = 0.5
    savi_denom = NIR + R + L
    savi = np.where(np.abs(savi_denom) > 1e-6,
                    ((NIR - R) / savi_denom) * (1.0 + L), 0.0)

    # ── NDBI (built-up proxy — using R as SWIR approximation) ─────────────────
    ndbi = _safe_ratio(R, NIR)

    # ── Brightness ────────────────────────────────────────────────────────────
    bright = (R + G + B + NIR) / 4.0

    # ── Aggregate: mean + std per index ───────────────────────────────────────
    def _ms(arr):
        return float(np.mean(arr)), float(np.std(arr))

    feats = []
    for arr in [ndvi, ndwi, evi, savi, ndbi, bright]:
        m, s = _ms(arr)
        feats.extend([m, s])

    return np.array(feats, dtype=np.float32)


def compute_spectral_embeddings(
    chips4: np.ndarray,
    progress_callback=None,
) -> np.ndarray:
    """
    Compute spectral feature vectors for all chips.

    Parameters
    ----------
    chips4 : np.ndarray  shape (N, 4, H, W)  float32 [0, 1]
    progress_callback : callable(done, total) | None

    Returns
    -------
    embs : np.ndarray  shape (N, SPECTRAL_DIM)  L2-normalized float32
    """
    N = chips4.shape[0]
    embs = np.zeros((N, SPECTRAL_DIM), dtype=np.float32)

    for i in range(N):
        embs[i] = compute_chip_spectral(chips4[i])
        if progress_callback and (i % 50 == 0 or i == N - 1):
            progress_callback(i + 1, N)

    faiss.normalize_L2(embs)
    return embs


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_spectral_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """Cosine similarity index over L2-normalized spectral vectors."""
    index = faiss.IndexFlatIP(SPECTRAL_DIM)
    index.add(embs)
    return index


def query_spectral_index(
    index: faiss.IndexFlatIP,
    embs: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """
    Query by an arbitrary L2-normalized spectral vector.

    Parameters
    ----------
    query_vec : np.ndarray  shape (SPECTRAL_DIM,) or (1, SPECTRAL_DIM)
    """
    qv = query_vec.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(qv)
    distances, indices = index.search(qv, top_k)
    return [int(i) for i in indices[0]], [float(s) for s in distances[0]]


def query_spectral_by_chip(
    index: faiss.IndexFlatIP,
    embs: np.ndarray,
    query_idx: int,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """Find chips with similar spectral signature to chip at query_idx."""
    query_vec = embs[query_idx : query_idx + 1].copy()
    distances, indices = index.search(query_vec, top_k + 1)
    result_indices, result_scores = [], []
    for idx, score in zip(indices[0], distances[0]):
        if idx == query_idx:
            continue
        result_indices.append(int(idx))
        result_scores.append(float(score))
        if len(result_indices) == top_k:
            break
    return result_indices, result_scores


# ── Persistence ───────────────────────────────────────────────────────────────

def save_spectral_index(index: faiss.IndexFlatIP, path: Path):
    faiss.write_index(index, str(path))


def load_spectral_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


def save_spectral_embeddings(embs: np.ndarray, path: Path):
    np.save(str(path), embs)


def load_spectral_embeddings(path: Path) -> np.ndarray:
    return np.load(str(path))


# ── Description → spectral query vector ──────────────────────────────────────
# Maps natural-language land cover concepts to synthetic spectral signatures.
# Values are scaled to [0,1] to match the normalised index space before L2 norm.

_SPECTRAL_ARCHETYPES = {
    # (ndvi_m, ndvi_s, ndwi_m, ndwi_s, evi_m, evi_s,
    #  savi_m, savi_s, ndbi_m, ndbi_s, bright_m, bright_s)
    "dense vegetation":     [0.75, 0.05, -0.50, 0.05, 0.55, 0.05, 0.55, 0.05, -0.50, 0.05, 0.35, 0.05],
    "sparse vegetation":    [0.30, 0.10, -0.15, 0.08, 0.20, 0.08, 0.22, 0.08, -0.10, 0.08, 0.40, 0.08],
    "stressed vegetation":  [0.15, 0.12,  0.00, 0.10, 0.08, 0.10, 0.10, 0.10,  0.05, 0.10, 0.38, 0.10],
    "open water":           [-0.15, 0.05, 0.25, 0.06, -0.10, 0.05, -0.12, 0.05, -0.20, 0.05, 0.20, 0.05],
    "impervious surface":   [-0.10, 0.08, -0.10, 0.08,  0.00, 0.08,  0.00, 0.08,  0.30, 0.08, 0.55, 0.10],
    "bare soil":            [0.05, 0.08, -0.05, 0.08,  0.03, 0.08,  0.04, 0.08,  0.20, 0.08, 0.50, 0.10],
    "urban":                [-0.05, 0.10, -0.08, 0.08,  0.00, 0.10,  0.00, 0.10,  0.35, 0.10, 0.60, 0.12],
    "agricultural":         [0.55, 0.10, -0.30, 0.08, 0.40, 0.10, 0.40, 0.10, -0.30, 0.08, 0.38, 0.08],
    "wetland":              [0.35, 0.12,  0.15, 0.10, 0.25, 0.10, 0.28, 0.10, -0.15, 0.08, 0.28, 0.08],
    "forest":               [0.80, 0.05, -0.55, 0.05, 0.60, 0.05, 0.60, 0.05, -0.55, 0.05, 0.30, 0.05],
    "grassland":            [0.40, 0.12, -0.20, 0.08, 0.28, 0.10, 0.30, 0.10, -0.18, 0.08, 0.42, 0.08],
    "parking lot":          [-0.08, 0.05, -0.08, 0.05, 0.00, 0.05,  0.00, 0.05,  0.38, 0.06, 0.58, 0.08],
    "rooftop":              [-0.05, 0.08, -0.06, 0.06, 0.00, 0.08,  0.00, 0.08,  0.32, 0.10, 0.65, 0.12],
    "high ndvi":            [0.75, 0.05, -0.50, 0.05, 0.55, 0.05, 0.55, 0.05, -0.50, 0.05, 0.35, 0.05],
    "low ndvi":             [-0.05, 0.08, -0.05, 0.08, 0.00, 0.08,  0.00, 0.08,  0.30, 0.08, 0.55, 0.10],
    "high ndwi":            [-0.20, 0.05, 0.30, 0.06, -0.12, 0.05, -0.15, 0.05, -0.25, 0.05, 0.18, 0.05],
    "high brightness":      [-0.02, 0.06, -0.05, 0.06,  0.00, 0.06,  0.00, 0.06,  0.35, 0.08, 0.70, 0.10],
    "low brightness":       [0.30, 0.10, -0.10, 0.08, 0.20, 0.08, 0.22, 0.08, -0.15, 0.08, 0.20, 0.06],
}


def concept_to_spectral_vector(concept: str) -> np.ndarray | None:
    """
    Return a synthetic L2-normalized spectral vector for a land cover concept,
    or None if no archetype matches.
    The concept string is lowercased and matched by substring.
    """
    concept_lc = concept.lower()
    best_key, best_overlap = None, 0
    for key in _SPECTRAL_ARCHETYPES:
        words = key.split()
        overlap = sum(1 for w in words if w in concept_lc)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key

    if best_key is None or best_overlap == 0:
        return None

    vec = np.array(_SPECTRAL_ARCHETYPES[best_key], dtype=np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def get_chip_spectral_report(chip4: np.ndarray) -> dict:
    """
    Return a human-readable spectral summary for a single chip,
    useful for display in the chat interface.
    """
    R, G, B, NIR = chip4[0], chip4[1], chip4[2], chip4[3]
    ndvi   = float(np.mean(_safe_ratio(NIR, R)))
    ndwi   = float(np.mean(_safe_ratio(G, NIR)))
    bright = float(np.mean((R + G + B + NIR) / 4.0))
    evi_n  = NIR - R
    evi_d  = NIR + 6.0 * R - 7.5 * B + 1.0
    evi    = float(np.mean(np.where(np.abs(evi_d) > 1e-6, 2.5 * evi_n / evi_d, 0.0)))

    def classify_ndvi(v):
        if v > 0.6:  return "dense vegetation"
        if v > 0.3:  return "moderate vegetation"
        if v > 0.1:  return "sparse/stressed vegetation"
        if v > 0.0:  return "bare soil / sparse cover"
        return "non-vegetated (water or impervious)"

    def classify_ndwi(v):
        if v > 0.2:  return "likely open water"
        if v > 0.0:  return "moist surface"
        return "dry surface"

    return {
        "ndvi":      round(ndvi, 3),
        "ndwi":      round(ndwi, 3),
        "evi":       round(evi,  3),
        "brightness": round(bright, 3),
        "ndvi_class":  classify_ndvi(ndvi),
        "ndwi_class":  classify_ndwi(ndwi),
    }
