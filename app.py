"""
NAIP Intelligence Platform
===========================
Tab 1 — 💬 VLM Chat     : fetch a NAIP tile by lat/lon and chat with Qwen3-VL
Tab 2 — 🔍 Similarity   : draw an AOI, chip the scene, embed with ResNet-50,
                           search visually similar locations via FAISS

Data: USDA NAIP via Microsoft Planetary Computer
"""

import io
import sys
import logging
import traceback
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image

from streamlit_folium import st_folium

import config
from utils.imagery import (
    search_naip_scenes,
    load_naip_scene,
    chip_scene,
    build_chip_geodataframe,
    cache_path,
    save_chips, load_chips,
    save_meta, load_meta,
)
from utils.embeddings import (
    load_model,
    embed_chips,
    build_index,
    query_index,
    save_index, load_index,
    save_embeddings, load_embeddings,
    umap_project,
)
from utils.viz import build_draw_map, build_result_map, build_umap_scatter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("naip_platform")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"]          { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3, .mono                  { font-family: 'Space Mono', monospace; }
  .stApp                             { background: #0d1117; color: #c9d1d9; }
  div[data-testid="stSidebar"]       { background: #161b22; border-right: 1px solid #21262d; }
  div[data-testid="stSidebar"] h3    { color: #58a6ff; }

  .badge {
    display: inline-block;
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 4px; padding: 2px 10px;
    font-size: 0.72rem; font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em; margin-right: 4px;
  }
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem; letter-spacing: 0.15em;
    color: #58a6ff; text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px; margin: 1.5rem 0 0.75rem 0;
  }
  .chip-caption {
    font-size: 0.7rem; color: #8b949e;
    font-family: 'Space Mono', monospace; text-align: center;
    margin-top: -8px;
  }
  .metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
  }
  .metric-card .val { font-size: 1.4rem; font-weight: 700; color: #e6edf3; }
  .metric-card .lbl { font-size: 0.72rem; color: #8b949e; }

  .stButton > button {
    background: #21262d; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    transition: all 0.15s ease;
  }
  .stButton > button:hover { background: #238636; color: #fff; border-color: #238636; }
  div[data-testid="stExpander"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
  }
  .bbox-display {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 6px 12px; font-family: 'Space Mono', monospace;
    font-size: 0.75rem; color: #58a6ff; margin-top: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛰️ NAIP Intelligence Platform")
st.markdown(
    '<span class="badge">PLANETARY COMPUTER</span>'
    '<span class="badge">QWEN3-VL</span>'
    '<span class="badge">RESNET-50</span>'
    '<span class="badge">FAISS COSINE</span>'
    '<span class="badge">OPEN SOURCE</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED SIDEBAR — credentials + common settings
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # --- Ollama credentials (used by Chat tab) ---
    st.markdown('<p class="section-header">Ollama Cloud</p>', unsafe_allow_html=True)

    # Prefer st.secrets, fall back to config defaults (env vars)
    def _secret(key, default=""):
        try:
            return st.secrets[key]
        except Exception:
            return default

    ollama_host  = st.text_input("Host URL",  value=_secret("OLLAMA_HOST",  config.OLLAMA_HOST_DEFAULT),  type="default")
    ollama_key   = st.text_input("API Key",   value=_secret("OLLAMA_API_KEY", config.OLLAMA_KEY_DEFAULT), type="password")
    ollama_model = st.text_input("Model",     value=_secret("OLLAMA_MODEL",  config.OLLAMA_MODEL_DEFAULT))

    st.markdown("---")
    st.caption(
        "Data: USDA NAIP via Microsoft Planetary Computer  \n"
        "Chat model: Ollama Cloud VLM  \n"
        "Embed model: ResNet-50 (ImageNet)  \n"
        "Index: FAISS IndexFlatIP (cosine)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_sim = st.tabs(["💬 VLM Chat", "🔍 Similarity Search"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — VLM CHAT  (ported from naipchat)
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<p class="section-header">00 · Area of Interest</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        lat = st.number_input("Latitude",  value=config.DEFAULT_LAT, format="%.4f", key="chat_lat")
    with c2:
        lon = st.number_input("Longitude", value=config.DEFAULT_LON, format="%.4f", key="chat_lon")
    with c3:
        buf = st.slider("Buffer (°)", 0.001, 0.01, 0.003, step=0.001, key="chat_buf")
    with c4:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        fetch_btn = st.button("🔍 Fetch NAIP Tile", key="chat_fetch")

    system_prompt = st.expander("🧠 System Prompt (click to edit)", expanded=False)
    with system_prompt:
        sys_prompt_val = st.text_area(
            label="system_prompt",
            label_visibility="collapsed",
            height=180,
            value=config.DEFAULT_SYSTEM_PROMPT,
            key="chat_sysprompt",
        )

    # ── NAIP fetch ────────────────────────────────────────────────────────────
    def fetch_naip_chat(lat, lon, buf):
        catalog = pystac_client.Client.open(
            config.PC_STAC_URL,
            modifier=pc.sign_inplace,
        )
        bbox = [lon - buf, lat - buf, lon + buf, lat + buf]
        results = catalog.search(
            collections=[config.NAIP_COLLECTION],
            bbox=bbox,
            limit=1,
            sortby="-properties.datetime",
        )
        items = list(results.items())
        if not items:
            raise ValueError("No NAIP tiles found for this location.")

        href = items[0].assets["image"].href
        with rasterio.open(href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read([1, 2, 3], window=window)

        data = np.moveaxis(data, 0, -1)
        data = np.clip(data, 0, 255).astype(np.uint8)
        img = Image.fromarray(data)

        buf_io = io.BytesIO()
        img.save(buf_io, format="PNG")
        buf_io.seek(0)
        b64 = base64.b64encode(buf_io.read()).decode("utf-8")
        return img, b64

    # ── Session state ─────────────────────────────────────────────────────────
    for k, v in [("chat_messages", []), ("chat_img", None), ("chat_b64", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    if fetch_btn:
        with st.spinner("Fetching NAIP from Planetary Computer…"):
            try:
                img, b64 = fetch_naip_chat(lat, lon, buf)
                st.session_state.chat_img = img
                st.session_state.chat_b64 = b64
                st.session_state.chat_messages = []
                st.success(f"Loaded — {img.width}×{img.height} px")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">01 · Image & Conversation</p>', unsafe_allow_html=True)
    img_col, chat_col = st.columns([1, 1])

    with img_col:
        st.subheader("🗺️ NAIP Tile")
        if st.session_state.chat_img:
            st.image(st.session_state.chat_img, use_container_width=True)
        else:
            st.info("Enter coordinates above and click **Fetch NAIP Tile**.")

    with chat_col:
        st.subheader("💬 Ask the VLM")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(
            "Ask about this image…",
            disabled=not st.session_state.chat_b64,
            key="chat_input",
        ):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if not ollama_host or not ollama_key:
                st.error("Set Ollama Host and API Key in the sidebar.")
            else:
                client = OpenAI(
                    base_url=f"{ollama_host.rstrip('/')}/v1",
                    api_key=ollama_key,
                )

                # Build message list: image injected on first user turn only
                openai_msgs = [{"role": "system", "content": sys_prompt_val}]
                for i, m in enumerate(st.session_state.chat_messages):
                    if i == 0:
                        openai_msgs.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": m["content"]},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{st.session_state.chat_b64}"
                                }},
                            ],
                        })
                    else:
                        openai_msgs.append({"role": m["role"], "content": m["content"]})

                with st.chat_message("assistant"):
                    box = st.empty()
                    full = ""
                    try:
                        stream = client.chat.completions.create(
                            model=ollama_model,
                            messages=openai_msgs,
                            stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            full += delta
                            box.markdown(full + "▌")
                        box.markdown(full)
                    except Exception as e:
                        st.error(f"Model error: {e}")

                st.session_state.chat_messages.append({"role": "assistant", "content": full})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SIMILARITY SEARCH  (ported from openembed)
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:

    # ── Sub-sidebar params injected inline ───────────────────────────────────
    sim_col_main, sim_col_ctrl = st.columns([4, 1])

    with sim_col_ctrl:
        st.markdown("**Imagery**")
        year = st.selectbox("NAIP Year", config.NAIP_YEARS, index=2, key="sim_year")

        st.markdown("**Chipping**")
        chip_size   = st.select_slider("Chip size (px)", [112, 224, 336], value=224, key="sim_chip")
        stride_frac = st.slider("Stride (fraction)", 0.25, 1.0, 0.5, 0.25, key="sim_stride")
        stride      = int(chip_size * stride_frac)

        st.markdown("**Search**")
        top_k     = st.slider("Top-K", 3, 20, config.DEFAULT_TOP_K, key="sim_topk")
        show_umap = st.checkbox("Show UMAP", value=True, key="sim_umap")

        # Session state init
        _SIM_KEYS = ["sim_scenes", "sim_selected", "sim_ds", "sim_chips",
                     "sim_positions", "sim_embeddings", "sim_faiss",
                     "sim_chip_gdf", "sim_query_idx", "sim_results",
                     "sim_umap_proj", "sim_bbox", "sim_scene_id"]
        for k in _SIM_KEYS:
            if k not in st.session_state:
                st.session_state[k] = None

        if st.session_state.sim_bbox:
            w, s, e, n = st.session_state.sim_bbox
            st.markdown(
                f'<div class="bbox-display">W {w:.4f}  E {e:.4f}<br>S {s:.4f}  N {n:.4f}</div>',
                unsafe_allow_html=True,
            )
            if (e - w) * (n - s) > 0.5:
                st.warning("⚠️ Large AOI")
        else:
            st.caption("🖊️ Draw a rectangle on the map.")

        search_btn = st.button("🔍 Search Scenes",  use_container_width=True,
                               disabled=(st.session_state.sim_bbox is None), key="sim_search")
        build_btn  = st.button("⚙️ Build Index",    use_container_width=True,
                               disabled=(st.session_state.sim_scenes is None), key="sim_build")

    with sim_col_main:
        # ── Step 0: Draw AOI ──────────────────────────────────────────────────
        st.markdown('<p class="section-header">00 · Draw Your Area of Interest</p>',
                    unsafe_allow_html=True)
        st.caption("Use the rectangle tool (top-left) to draw your AOI, then click **Search Scenes**.")

        if st.session_state.sim_bbox:
            w, s, e, n = st.session_state.sim_bbox
            map_center = ((s + n) / 2, (w + e) / 2)
            map_zoom   = 12
        else:
            db = config.DEFAULT_BBOX
            map_center = (
                db["south"] + (db["north"] - db["south"]) / 2,
                db["west"]  + (db["east"]  - db["west"])  / 2,
            )
            map_zoom = 10

        draw_map = build_draw_map(
            center=map_center,
            zoom=map_zoom,
            existing_bbox=st.session_state.sim_bbox,
        )
        map_data = st_folium(draw_map, width="100%", height=400,
                             returned_objects=["all_drawings"], key="sim_aoi_map")

        # Parse drawn rectangle
        def _extract_bbox(md):
            try:
                drawings = md.get("all_drawings") or []
                if not drawings:
                    return None
                geom = drawings[-1].get("geometry", {})
                if geom.get("type") != "Polygon":
                    return None
                coords = geom["coordinates"][0]
                lngs = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                return [min(lngs), min(lats), max(lngs), max(lats)]
            except Exception:
                return None

        drawn = _extract_bbox(map_data)
        if drawn:
            w, s, e, n = drawn
            if (e - w) > 0 and (n - s) > 0 and drawn != st.session_state.sim_bbox:
                st.session_state.sim_bbox = drawn
                for k in ["sim_scenes", "sim_selected", "sim_ds", "sim_chips",
                          "sim_positions", "sim_embeddings", "sim_faiss",
                          "sim_chip_gdf", "sim_query_idx", "sim_results",
                          "sim_umap_proj", "sim_scene_id"]:
                    st.session_state[k] = None
                st.rerun()

        if st.session_state.sim_bbox:
            w, s, e, n = st.session_state.sim_bbox
            st.success(f"✅ AOI — W:{w:.4f} S:{s:.4f} E:{e:.4f} N:{n:.4f}")
        else:
            st.info("No AOI drawn yet.")

    # ── Step 1: Scene search ──────────────────────────────────────────────────
    if search_btn:
        bbox = st.session_state.sim_bbox
        if not bbox:
            st.error("Draw an AOI first.")
            st.stop()
        with st.spinner("📡 Querying Planetary Computer STAC…"):
            try:
                scenes = search_naip_scenes(bbox, year)
            except Exception as e:
                st.error(f"STAC search failed: {e}")
                log.error(traceback.format_exc())
                st.stop()
        if not scenes:
            st.error(f"No NAIP scenes found for this AOI in {year}. Try a different year or larger extent.")
            st.stop()
        st.session_state.sim_scenes = scenes
        for k in ["sim_ds", "sim_chips", "sim_positions", "sim_embeddings", "sim_faiss",
                  "sim_chip_gdf", "sim_query_idx", "sim_results", "sim_umap_proj", "sim_scene_id"]:
            st.session_state[k] = None
        st.success(f"Found **{len(scenes)}** scene(s). Select one below and click **Build Index**.")

    # ── Scene picker ──────────────────────────────────────────────────────────
    if st.session_state.sim_scenes:
        st.markdown('<p class="section-header">01 · Select Scene</p>', unsafe_allow_html=True)
        labels = [
            f"{i+1}. {s.id}  |  {s.datetime.date() if s.datetime else 'n/a'}"
            for i, s in enumerate(st.session_state.sim_scenes)
        ]
        sel = st.radio("Scenes", labels, label_visibility="collapsed", key="sim_radio")
        st.session_state.sim_selected = st.session_state.sim_scenes[labels.index(sel)]

        item = st.session_state.sim_selected
        with st.expander("Scene metadata"):
            st.dataframe(pd.DataFrame({
                "Field": ["ID", "Date", "CRS", "Cloud Cover", "State"],
                "Value": [
                    item.id,
                    str(item.datetime.date()) if item.datetime else "n/a",
                    item.properties.get("proj:epsg", "n/a"),
                    item.properties.get("eo:cloud_cover", "n/a"),
                    item.properties.get("naip:state", "n/a"),
                ],
            }), use_container_width=True, hide_index=True)

    # ── Cached model ──────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def _load_resnet():
        return load_model()

    # ── Step 2: Build index ───────────────────────────────────────────────────
    if build_btn and st.session_state.sim_selected is not None:
        item     = st.session_state.sim_selected
        scene_id = item.id

        chips_path = cache_path(scene_id, chip_size, stride, "_chips.npy")
        embs_path  = cache_path(scene_id, chip_size, stride, "_embeddings.npy")
        idx_path   = cache_path(scene_id, chip_size, stride, "_faiss.index")
        meta_path  = cache_path(scene_id, chip_size, stride, "_meta.pkl")
        cache_hit  = all(p.exists() for p in [chips_path, embs_path, idx_path, meta_path])

        if cache_hit:
            st.info("💾 Loading from disk cache…")
            chips     = load_chips(chips_path)
            embs      = load_embeddings(embs_path)
            faiss_idx = load_index(idx_path)
            cached    = load_meta(meta_path)
            positions = cached["positions"]
            chip_gdf  = cached["chip_gdf"]
        else:
            with st.spinner("📥 Loading NAIP COG…"):
                try:
                    ds, ov_level = load_naip_scene(item)
                except Exception as e:
                    st.error(f"Failed to load imagery: {e}")
                    log.error(traceback.format_exc())
                    st.stop()

            st.success(f"Loaded `{scene_id}` — {ds.shape[1]}×{ds.shape[2]} px (overview {ov_level})")

            with st.spinner("✂️ Chipping imagery…"):
                chips, positions = chip_scene(ds, chip_size=chip_size, stride=stride)

            if len(chips) == 0:
                st.error("No chips generated — AOI may be smaller than one chip. Reduce chip size or draw a larger AOI.")
                st.stop()
            if len(chips) >= config.MAX_CHIPS:
                st.warning(f"⚠️ Chip count capped at {config.MAX_CHIPS}.")

            st.info(f"Generated **{len(chips)}** chips  ({chip_size}px, stride {stride}px)")

            with st.spinner("📐 Building chip GeoDataFrame…"):
                chip_gdf = build_chip_geodataframe(positions, ds, chip_size)

            model, device = _load_resnet()
            progress_bar  = st.progress(0.0, text="🧠 Embedding chips…")

            def _cb(done, total):
                progress_bar.progress(done / total, text=f"🧠 {done}/{total} chips…")

            try:
                embs = embed_chips(chips, model, device, progress_callback=_cb)
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                log.error(traceback.format_exc())
                st.stop()
            finally:
                progress_bar.empty()

            with st.spinner("🗂️ Building FAISS index…"):
                faiss_idx = build_index(embs)

            with st.spinner("💾 Saving to cache…"):
                save_chips(chips, chips_path)
                save_embeddings(embs, embs_path)
                save_index(faiss_idx, idx_path)
                save_meta({"positions": positions, "chip_gdf": chip_gdf}, meta_path)

        st.session_state.sim_chips      = chips
        st.session_state.sim_positions  = positions
        st.session_state.sim_embeddings = embs
        st.session_state.sim_faiss      = faiss_idx
        st.session_state.sim_chip_gdf   = chip_gdf
        st.session_state.sim_scene_id   = scene_id
        st.session_state.sim_query_idx  = None
        st.session_state.sim_results    = None
        st.session_state.sim_umap_proj  = None

        mc = st.columns(4)
        mc[0].markdown(f'<div class="metric-card"><div class="val">{len(chips):,}</div><div class="lbl">Chips</div></div>', unsafe_allow_html=True)
        mc[1].markdown(f'<div class="metric-card"><div class="val">{embs.shape[1]:,}</div><div class="lbl">Embed dim</div></div>', unsafe_allow_html=True)
        mc[2].markdown(f'<div class="metric-card"><div class="val">{chip_size}px</div><div class="lbl">Chip size</div></div>', unsafe_allow_html=True)
        mc[3].markdown(f'<div class="metric-card"><div class="val">{"cache ✓" if cache_hit else "fresh"}</div><div class="lbl">Source</div></div>', unsafe_allow_html=True)

    # ── Step 3: Query ─────────────────────────────────────────────────────────
    if st.session_state.sim_chips is not None:
        chips     = st.session_state.sim_chips
        n_chips   = len(chips)
        embs      = st.session_state.sim_embeddings
        faiss_idx = st.session_state.sim_faiss
        chip_gdf  = st.session_state.sim_chip_gdf

        st.markdown("---")
        st.markdown('<p class="section-header">02 · Select Query Chip</p>', unsafe_allow_html=True)

        qc1, qc2, qc3 = st.columns([2, 1, 3])
        with qc1:
            current_q = st.session_state.sim_query_idx if st.session_state.sim_query_idx is not None else 0
            query_idx = st.number_input(
                f"Chip index (0–{n_chips - 1})",
                min_value=0, max_value=n_chips - 1,
                value=current_q, step=1, key="sim_q_input",
            )
        with qc2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("🎲 Random", key="sim_random"):
                st.session_state.sim_query_idx = int(np.random.randint(0, n_chips))
                st.rerun()
        with qc3:
            chip_rgb = (chips[query_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            st.image(chip_rgb, caption=f"Query — chip #{query_idx}", width=180)

        with st.expander("🖼️ Browse chips (sample)"):
            sample_n   = min(24, n_chips)
            step_size  = max(1, n_chips // sample_n)
            sample_ids = list(range(0, n_chips, step_size))[:sample_n]
            gcols      = st.columns(8)
            for j, sid in enumerate(sample_ids):
                rgb = (chips[sid].transpose(1, 2, 0) * 255).astype(np.uint8)
                gcols[j % 8].image(rgb, use_container_width=True)
                gcols[j % 8].markdown(f'<p class="chip-caption">#{sid}</p>', unsafe_allow_html=True)

        find_btn = st.button(f"🔎 Find top-{top_k} similar chips", type="primary", key="sim_find")

        if find_btn:
            result_indices, result_scores = query_index(faiss_idx, embs, query_idx, top_k)
            st.session_state.sim_query_idx = query_idx
            st.session_state.sim_results   = (result_indices, result_scores)

            if show_umap and st.session_state.sim_umap_proj is None:
                with st.spinner("📐 Computing UMAP projection… (~30 s one-time)"):
                    try:
                        st.session_state.sim_umap_proj = umap_project(embs)
                    except ImportError:
                        st.warning("Install `umap-learn` for UMAP visualization.")
                    except Exception as e:
                        st.warning(f"UMAP failed: {e}")

    # ── Step 4: Results ───────────────────────────────────────────────────────
    if st.session_state.sim_results is not None:
        result_indices, result_scores = st.session_state.sim_results
        query_idx = st.session_state.sim_query_idx
        chip_gdf  = st.session_state.sim_chip_gdf
        chips     = st.session_state.sim_chips
        bbox_used = st.session_state.sim_bbox

        st.markdown("---")
        st.markdown(f'<p class="section-header">03 · Top-{len(result_indices)} Similar Chips</p>',
                    unsafe_allow_html=True)

        n_cols   = min(len(result_indices), 8)
        res_cols = st.columns(n_cols)
        for col, (idx, score) in zip(res_cols, zip(result_indices, result_scores)):
            rgb = (chips[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            col.image(rgb, use_container_width=True)
            col.markdown(f'<p class="chip-caption">#{idx}<br>{score:.4f}</p>', unsafe_allow_html=True)

        tab_map, tab_umap, tab_table, tab_export = st.tabs(
            ["🗺️ Map", "📐 Embedding Space", "📊 Table", "💾 Export"]
        )

        with tab_map:
            center = (
                ((bbox_used[1] + bbox_used[3]) / 2, (bbox_used[0] + bbox_used[2]) / 2)
                if bbox_used
                else (chip_gdf.dissolve().centroid.iloc[0].y,
                      chip_gdf.dissolve().centroid.iloc[0].x)
            )
            fmap = build_result_map(query_idx, result_indices, result_scores, chip_gdf, center)
            st_folium(fmap, width="100%", height=520, returned_objects=[], key="sim_result_map")

        with tab_umap:
            if st.session_state.sim_umap_proj is not None:
                fig = build_umap_scatter(
                    st.session_state.sim_umap_proj,
                    query_idx, result_indices, result_scores, len(chips),
                )
                st.plotly_chart(fig, use_container_width=True)
            elif show_umap:
                st.info("UMAP not yet computed. Click **Find similar chips** to trigger it.")
            else:
                st.info("Enable **Show UMAP** above, then run a search.")

        with tab_table:
            df = pd.DataFrame({
                "Rank":       range(1, len(result_indices) + 1),
                "Chip ID":    result_indices,
                "Cosine Sim": [f"{s:.6f}" for s in result_scores],
                "Pixel Row":  [chip_gdf.loc[chip_gdf["chip_id"] == i, "pixel_row"].iloc[0] for i in result_indices],
                "Pixel Col":  [chip_gdf.loc[chip_gdf["chip_id"] == i, "pixel_col"].iloc[0] for i in result_indices],
            })
            st.dataframe(df.set_index("Rank"), use_container_width=True)

        with tab_export:
            st.markdown("#### Download results")
            result_gdf = chip_gdf[chip_gdf["chip_id"].isin(result_indices)].copy()
            score_map  = dict(zip(result_indices, result_scores))
            result_gdf["cosine_sim"] = result_gdf["chip_id"].map(score_map)
            result_gdf["query_chip"] = query_idx
            result_gdf["scene_id"]   = st.session_state.sim_scene_id

            st.download_button(
                "⬇️ GeoJSON — similar chip polygons",
                data=result_gdf.to_json().encode(),
                file_name=f"naip_similar_{st.session_state.sim_scene_id}.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
            df_exp = pd.DataFrame({"Rank": range(1, len(result_indices)+1),
                                   "Chip ID": result_indices,
                                   "Cosine Sim": result_scores})
            st.download_button(
                "⬇️ CSV — similarity scores",
                data=df_exp.to_csv(index=False).encode(),
                file_name=f"naip_scores_{st.session_state.sim_scene_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            buf = io.BytesIO()
            np.save(buf, st.session_state.sim_embeddings[[query_idx] + result_indices])
            st.download_button(
                "⬇️ NPY — embeddings (query + results)",
                data=buf.getvalue(),
                file_name=f"naip_embeddings_{st.session_state.sim_scene_id}.npy",
                mime="application/octet-stream",
                use_container_width=True,
            )
            st.caption(
                "GeoJSON loads directly into QGIS, ArcGIS Pro, or GeoPandas.  \n"
                "NPY: L2-normalized float32, shape (K+1, 2048)."
            )
