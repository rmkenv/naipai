"""
utils/imagery.py — NAIP fetch, chipping, and geo-projection helpers
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pystac_client
import planetary_computer
import rioxarray
import geopandas as gpd
from shapely.geometry import box

import naip_config as cfg

PC_STAC_URL     = cfg.PC_STAC_URL
NAIP_COLLECTION = cfg.NAIP_COLLECTION
MAX_SCENES      = cfg.MAX_SCENES
OVERVIEW_LEVELS = cfg.OVERVIEW_LEVELS
DEFAULT_CHIP_SIZE = cfg.DEFAULT_CHIP_SIZE
MAX_CHIPS       = cfg.MAX_CHIPS
CACHE_DIR       = cfg.CACHE_DIR


def get_catalog():
    return pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def search_naip_scenes(bbox, year):
    catalog = get_catalog()
    search = catalog.search(
        collections=[NAIP_COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=MAX_SCENES,
    )
    return list(search.items())


def load_naip_scene(item, overview_level=None):
    href = item.assets["image"].href
    levels = [overview_level] if overview_level is not None else OVERVIEW_LEVELS
    for lvl in levels:
        try:
            ds = rioxarray.open_rasterio(href, overview_level=lvl)
            if ds.shape[1] > 0 and ds.shape[2] > 0:
                return ds, lvl
        except Exception:
            continue
    return rioxarray.open_rasterio(href), None


def normalize_rgb(arr):
    lo = np.percentile(arr, 2)
    hi = np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def chip_scene(ds, chip_size=DEFAULT_CHIP_SIZE, stride=None, max_chips=MAX_CHIPS):
    if stride is None:
        stride = chip_size // 2
    arr = normalize_rgb(ds.values[:3].astype(np.float32))
    _, H, W = arr.shape
    chips, positions = [], []
    for y in range(0, H - chip_size, stride):
        for x in range(0, W - chip_size, stride):
            chip = arr[:, y:y + chip_size, x:x + chip_size]
            if chip.shape == (3, chip_size, chip_size):
                chips.append(chip)
                positions.append((y, x))
            if len(chips) >= max_chips:
                break
        if len(chips) >= max_chips:
            break
    return np.stack(chips), positions


def chip_scene_4band(ds, chip_size=DEFAULT_CHIP_SIZE, stride=None, max_chips=MAX_CHIPS):
    if stride is None:
        stride = chip_size // 2
    n_bands = min(ds.values.shape[0], 4)
    arr = ds.values[:n_bands].astype(np.float32)
    arr_norm = np.zeros_like(arr)
    for b in range(n_bands):
        lo = np.percentile(arr[b], 2)
        hi = np.percentile(arr[b], 98)
        arr_norm[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-8), 0, 1)
    if n_bands < 4:
        pad = np.zeros((4 - n_bands, arr.shape[1], arr.shape[2]), dtype=np.float32)
        arr_norm = np.concatenate([arr_norm, pad], axis=0)
    _, H, W = arr_norm.shape
    chips, positions = [], []
    for y in range(0, H - chip_size, stride):
        for x in range(0, W - chip_size, stride):
            chip = arr_norm[:, y:y + chip_size, x:x + chip_size]
            if chip.shape == (4, chip_size, chip_size):
                chips.append(chip)
                positions.append((y, x))
            if len(chips) >= max_chips:
                break
        if len(chips) >= max_chips:
            break
    return np.stack(chips), positions


def pixel_to_bbox(pos, ds, chip_size):
    tf = ds.rio.transform()
    row, col = pos
    left  = tf.c + col * tf.a
    top   = tf.f + row * tf.e
    right = left + chip_size * tf.a
    bot   = top  + chip_size * tf.e
    return box(left, bot, right, top)


def build_chip_geodataframe(positions, ds, chip_size, crs="EPSG:4326"):
    scene_crs = ds.rio.crs or "EPSG:4326"
    geoms = [pixel_to_bbox(p, ds, chip_size) for p in positions]
    gdf = gpd.GeoDataFrame(
        {"chip_id": range(len(geoms)),
         "pixel_row": [p[0] for p in positions],
         "pixel_col": [p[1] for p in positions]},
        geometry=geoms, crs=scene_crs,
    )
    if str(scene_crs) != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def _cache_key(scene_id, chip_size, stride):
    return hashlib.md5(f"{scene_id}_{chip_size}_{stride}".encode()).hexdigest()[:12]


def cache_path(scene_id, chip_size, stride, suffix):
    return CACHE_DIR / f"{_cache_key(scene_id, chip_size, stride)}{suffix}"


def save_chips(chips, path):   np.save(str(path), chips)
def load_chips(path):          return np.load(str(path))
def save_meta(meta, path):
    with open(path, "wb") as f: pickle.dump(meta, f)
def load_meta(path):
    with open(path, "rb") as f: return pickle.load(f)
