"""
utils/spectral.py — Spectral feature extraction and FAISS index for NAIP RGBI chips.

NAIP band order: 1=Red  2=Green  3=Blue  4=NIR

12-d feature vector per chip (mean+std of 6 indices):
  NDVI, NDWI, EVI, SAVI, NDBI, Brightness
"""
from pathlib import Path
import numpy as np
import faiss

import naip_config as cfg

CACHE_DIR = cfg.CACHE_DIR

SPECTRAL_INDEX_NAMES = [
    "ndvi_mean", "ndvi_std",
    "ndwi_mean", "ndwi_std",
    "evi_mean",  "evi_std",
    "savi_mean", "savi_std",
    "ndbi_mean", "ndbi_std",
    "brightness_mean", "brightness_std",
]
SPECTRAL_DIM = len(SPECTRAL_INDEX_NAMES)


def _safe_ratio(a, b):
    denom = a + b
    return np.where(np.abs(denom) > 1e-6, (a - b) / denom, 0.0)


def compute_chip_spectral(chip4):
    R, G, B, NIR = chip4[0], chip4[1], chip4[2], chip4[3]
    ndvi   = _safe_ratio(NIR, R)
    ndwi   = _safe_ratio(G, NIR)
    evi_n  = NIR - R
    evi_d  = NIR + 6.0 * R - 7.5 * B + 1.0
    evi    = np.clip(np.where(np.abs(evi_d) > 1e-6, 2.5 * evi_n / evi_d, 0.0), -2.0, 2.0)
    L      = 0.5
    savi_d = NIR + R + L
    savi   = np.where(np.abs(savi_d) > 1e-6, ((NIR - R) / savi_d) * (1.0 + L), 0.0)
    ndbi   = _safe_ratio(R, NIR)
    bright = (R + G + B + NIR) / 4.0
    feats  = []
    for arr in [ndvi, ndwi, evi, savi, ndbi, bright]:
        feats.extend([float(np.mean(arr)), float(np.std(arr))])
    return np.array(feats, dtype=np.float32)


def compute_spectral_embeddings(chips4, progress_callback=None):
    N    = chips4.shape[0]
    embs = np.zeros((N, SPECTRAL_DIM), dtype=np.float32)
    for i in range(N):
        embs[i] = compute_chip_spectral(chips4[i])
        if progress_callback and (i % 50 == 0 or i == N - 1):
            progress_callback(i + 1, N)
    faiss.normalize_L2(embs)
    return embs


def build_spectral_index(embs):
    index = faiss.IndexFlatIP(SPECTRAL_DIM)
    index.add(embs)
    return index


def query_spectral_index(index, embs, query_vec, top_k):
    qv = query_vec.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(qv)
    distances, indices = index.search(qv, top_k)
    return [int(i) for i in indices[0]], [float(s) for s in distances[0]]


def query_spectral_by_chip(index, embs, query_idx, top_k):
    qv = embs[query_idx:query_idx + 1].copy()
    distances, indices = index.search(qv, top_k + 1)
    result_indices, result_scores = [], []
    for idx, score in zip(indices[0], distances[0]):
        if idx == query_idx:
            continue
        result_indices.append(int(idx))
        result_scores.append(float(score))
        if len(result_indices) == top_k:
            break
    return result_indices, result_scores


def save_spectral_index(index, path):     faiss.write_index(index, str(path))
def load_spectral_index(path):            return faiss.read_index(str(path))
def save_spectral_embeddings(embs, path): np.save(str(path), embs)
def load_spectral_embeddings(path):       return np.load(str(path))


_SPECTRAL_ARCHETYPES = {
    "dense vegetation":    [0.75,0.05,-0.50,0.05,0.55,0.05,0.55,0.05,-0.50,0.05,0.35,0.05],
    "sparse vegetation":   [0.30,0.10,-0.15,0.08,0.20,0.08,0.22,0.08,-0.10,0.08,0.40,0.08],
    "stressed vegetation": [0.15,0.12, 0.00,0.10,0.08,0.10,0.10,0.10, 0.05,0.10,0.38,0.10],
    "open water":          [-0.15,0.05,0.25,0.06,-0.10,0.05,-0.12,0.05,-0.20,0.05,0.20,0.05],
    "impervious surface":  [-0.10,0.08,-0.10,0.08, 0.00,0.08, 0.00,0.08, 0.30,0.08,0.55,0.10],
    "bare soil":           [0.05,0.08,-0.05,0.08,  0.03,0.08, 0.04,0.08, 0.20,0.08,0.50,0.10],
    "urban":               [-0.05,0.10,-0.08,0.08, 0.00,0.10, 0.00,0.10, 0.35,0.10,0.60,0.12],
    "agricultural":        [0.55,0.10,-0.30,0.08,  0.40,0.10, 0.40,0.10,-0.30,0.08,0.38,0.08],
    "wetland":             [0.35,0.12, 0.15,0.10,  0.25,0.10, 0.28,0.10,-0.15,0.08,0.28,0.08],
    "forest":              [0.80,0.05,-0.55,0.05,  0.60,0.05, 0.60,0.05,-0.55,0.05,0.30,0.05],
    "grassland":           [0.40,0.12,-0.20,0.08,  0.28,0.10, 0.30,0.10,-0.18,0.08,0.42,0.08],
    "parking lot":         [-0.08,0.05,-0.08,0.05, 0.00,0.05, 0.00,0.05, 0.38,0.06,0.58,0.08],
    "rooftop":             [-0.05,0.08,-0.06,0.06, 0.00,0.08, 0.00,0.08, 0.32,0.10,0.65,0.12],
    "high ndvi":           [0.75,0.05,-0.50,0.05,  0.55,0.05, 0.55,0.05,-0.50,0.05,0.35,0.05],
    "low ndvi":            [-0.05,0.08,-0.05,0.08, 0.00,0.08, 0.00,0.08, 0.30,0.08,0.55,0.10],
    "high ndwi":           [-0.20,0.05,0.30,0.06, -0.12,0.05,-0.15,0.05,-0.25,0.05,0.18,0.05],
    "high brightness":     [-0.02,0.06,-0.05,0.06, 0.00,0.06, 0.00,0.06, 0.35,0.08,0.70,0.10],
    "low brightness":      [0.30,0.10,-0.10,0.08,  0.20,0.08, 0.22,0.08,-0.15,0.08,0.20,0.06],
}


def concept_to_spectral_vector(concept):
    concept_lc = concept.lower()
    best_key, best_overlap = None, 0
    for key in _SPECTRAL_ARCHETYPES:
        overlap = sum(1 for w in key.split() if w in concept_lc)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    if best_key is None or best_overlap == 0:
        return None
    vec = np.array(_SPECTRAL_ARCHETYPES[best_key], dtype=np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def get_chip_spectral_report(chip4):
    R, G, B, NIR = chip4[0], chip4[1], chip4[2], chip4[3]
    ndvi   = float(np.mean(_safe_ratio(NIR, R)))
    ndwi   = float(np.mean(_safe_ratio(G, NIR)))
    bright = float(np.mean((R + G + B + NIR) / 4.0))
    evi_n  = NIR - R
    evi_d  = NIR + 6.0 * R - 7.5 * B + 1.0
    evi    = float(np.mean(np.where(np.abs(evi_d) > 1e-6, 2.5 * evi_n / evi_d, 0.0)))

    def _ndvi_class(v):
        if v > 0.6:   return "dense vegetation"
        if v > 0.3:   return "moderate vegetation"
        if v > 0.1:   return "sparse/stressed vegetation"
        if v > 0.0:   return "bare soil / sparse cover"
        return "non-vegetated"

    def _ndwi_class(v):
        if v > 0.2:   return "likely open water"
        if v > 0.0:   return "moist surface"
        return "dry surface"

    return {
        "ndvi": round(ndvi, 3), "ndwi": round(ndwi, 3),
        "evi":  round(evi,  3), "brightness": round(bright, 3),
        "ndvi_class": _ndvi_class(ndvi),
        "ndwi_class": _ndwi_class(ndwi),
    }
