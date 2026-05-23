"""
utils/embeddings.py — ResNet-50 embedding pipeline + FAISS index management
"""
from pathlib import Path

import numpy as np
import faiss
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import naip_config as cfg

EMBED_DIM     = cfg.EMBED_DIM
BATCH_SIZE    = cfg.BATCH_SIZE
IMAGENET_MEAN = cfg.IMAGENET_MEAN
IMAGENET_STD  = cfg.IMAGENET_STD


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device=None):
    if device is None:
        device = get_device()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device), device


_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def embed_chips(chips, model, device, batch_size=BATCH_SIZE, progress_callback=None):
    all_embs = []
    total = len(chips)
    for start in range(0, total, batch_size):
        batch_np = chips[start:start + batch_size]
        batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
        batch = torch.stack([_normalize(b) for b in batch])
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        all_embs.append(emb)
        if progress_callback is not None:
            progress_callback(min(start + batch_size, total), total)
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    faiss.normalize_L2(embs)
    return embs


def build_index(embs):
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def query_index(index, embs, query_idx, top_k):
    query_vec = embs[query_idx:query_idx + 1]
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


def query_index_vec(index, query_vec, top_k):
    qv = query_vec.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(qv)
    distances, indices = index.search(qv, top_k)
    return [int(i) for i in indices[0]], [float(s) for s in distances[0]]


def save_index(index, path):      faiss.write_index(index, str(path))
def load_index(path):             return faiss.read_index(str(path))
def save_embeddings(embs, path):  np.save(str(path), embs)
def load_embeddings(path):        return np.load(str(path))
