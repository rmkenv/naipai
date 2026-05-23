"""
NAIP Intelligence Platform
===========================
One unified interface: ask anything about the imagery.
  — Describe / analyze / question  →  VLM answers conversationally
  — Find / locate / search for     →  ResNet-50 + FAISS similarity search
                                       highlights matching locations on the map

Data: USDA NAIP via Microsoft Planetary Computer
"""

import io, sys, logging, traceback, base64
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
import folium
from streamlit_folium import st_folium

import config
from utils.imagery import (
    search_naip_scenes, load_naip_scene, chip_scene,
    build_chip_geodataframe, cache_path,
    save_chips, load_chips, save_meta, load_meta,
)
from utils.embeddings import (
    load_model, embed_chips, build_index, query_index,
    save_index, load_index, save_embeddings, load_embeddings,
)
from utils.viz import build_result_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("naip_platform")

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"]   { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3                   { font-family: 'Space Mono', monospace; }
  .stApp                       { background: #0d1117; color: #c9d1d9; }
  div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }

  .badge {
    display: inline-block;
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55; border-radius: 4px;
    padding: 2px 10px; font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em; margin-right: 4px;
  }
  .intent-pill {
    display: inline-block; border-radius: 12px;
    padding: 2px 12px; font-size: 0.72rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.1em; margin-bottom: 6px;
  }
  .pill-chat   { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb66; }
  .pill-search { background: #23863633; color: #3fb950; border: 1px solid #23863666; }
  .result-chip { text-align: center; font-size: 0.7rem;
                 color: #8b949e; font-family: 'Space Mono', monospace; }
  .stButton > button {
    background: #21262d; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
  }
  .stButton > button:hover { background: #238636; color: #fff; border-color: #238636; }
  .hint-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 16px;
    font-size: 0.8rem; color: #8b949e; margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛰️ NAIP Intelligence Platform")
st.markdown(
    '<span class="badge">PLANETARY COMPUTER</span>'
    '<span class="badge">QWEN3-VL</span>'
    '<span class="badge">RESNET-50 · FAISS</span>'
    '<span class="badge">NYC DEFAULT</span>',
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
DEFAULTS = {
    "naip_img": None, "naip_b64": None, "naip_scene": None,
    "messages": [],
    "chips": None, "positions": None, "embeddings": None,
    "faiss_idx": None, "chip_gdf": None, "scene_id": None,
    "last_bbox": None,
    "result_map_html": None, "result_chips": None, "result_scores": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    def _secret(key, default=""):
        try:    return st.secrets[key]
        except: return default

    ollama_host  = st.text_input("Ollama Host",  value=_secret("OLLAMA_HOST",  config.OLLAMA_HOST_DEFAULT))
    ollama_key   = st.text_input("API Key",       value=_secret("OLLAMA_API_KEY", config.OLLAMA_KEY_DEFAULT), type="password")
    ollama_model = st.text_input("Model",         value=_secret("OLLAMA_MODEL",  config.OLLAMA_MODEL_DEFAULT))

    st.markdown("---")
    st.markdown("**Scene settings**")
    year      = st.selectbox("NAIP Year", config.NAIP_YEARS, index=2)
    chip_size = st.select_slider("Chip size (px)", [112, 224, 336], value=224)
    top_k     = st.slider("Results to find", 3, 20, config.DEFAULT_TOP_K)

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
<div class="hint-box">
Ask anything in the chat below.<br><br>
<b>To analyze:</b> "What land cover types are present?" or "Describe the road network."<br><br>
<b>To find:</b> "Find all parking lots" or "Locate green spaces" or "Show me rooftops."
</div>
""", unsafe_allow_html=True)

    st.caption("Data: USDA NAIP via Planetary Computer")

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ══════════════════════════════════════════════════════════════════════════════
def get_client():
    if not ollama_host or not ollama_key:
        return None
    return OpenAI(
        base_url=f"{ollama_host.rstrip('/')}/v1",
        api_key=ollama_key,
    )

def classify_intent(user_text: str) -> str:
    """Returns 'CHAT' or 'SEARCH'."""
    client = get_client()
    if not client:
        return "CHAT"
    try:
        resp = client.chat.completions.create(
            model=ollama_model,
            messages=[
                {"role": "system", "content": config.INTENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            max_tokens=5,
            temperature=0,
        )
        word = resp.choices[0].message.content.strip().upper()
        return "SEARCH" if "SEARCH" in word else "CHAT"
    except Exception:
        return "CHAT"

# ══════════════════════════════════════════════════════════════════════════════
# NAIP FETCH (point + buffer → RGB chip)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_naip_point(lat, lon, buf=0.003):
    catalog = pystac_client.Client.open(config.PC_STAC_URL, modifier=pc.sign_inplace)
    bbox = [lon - buf, lat - buf, lon + buf, lat + buf]
    results = catalog.search(
        collections=[config.NAIP_COLLECTION], bbox=bbox,
        limit=1, sortby="-properties.datetime",
    )
    items = list(results.items())
    if not items:
        raise ValueError("No NAIP scenes found at this location.")

    item = items[0]
    href = item.assets["image"].href
    with rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*bounds, transform=src.transform)
        data   = src.read([1, 2, 3], window=window)

    data = np.moveaxis(data, 0, -1)
    data = np.clip(data, 0, 255).astype(np.uint8)
    img  = Image.fromarray(data)

    buf_io = io.BytesIO()
    img.save(buf_io, format="PNG")
    buf_io.seek(0)
    b64 = base64.b64encode(buf_io.read()).decode("utf-8")
    return img, b64, item

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING INDEX (build or load from cache)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_resnet():
    return load_model()

def ensure_index(item, stride):
    scene_id = item.id
    if st.session_state.scene_id == scene_id and st.session_state.faiss_idx is not None:
        return  # already built for this scene

    chips_path = cache_path(scene_id, chip_size, stride, "_chips.npy")
    embs_path  = cache_path(scene_id, chip_size, stride, "_embeddings.npy")
    idx_path   = cache_path(scene_id, chip_size, stride, "_faiss.index")
    meta_path  = cache_path(scene_id, chip_size, stride, "_meta.pkl")
    cached     = all(p.exists() for p in [chips_path, embs_path, idx_path, meta_path])

    if cached:
        chips     = load_chips(chips_path)
        embs      = load_embeddings(embs_path)
        faiss_idx = load_index(idx_path)
        meta      = load_meta(meta_path)
    else:
        with st.spinner("📥 Loading full NAIP scene for indexing…"):
            ds, _ = load_naip_scene(item)

        with st.spinner(f"✂️ Chipping scene into {chip_size}px tiles…"):
            chips, positions = chip_scene(ds, chip_size=chip_size, stride=stride)

        with st.spinner("📐 Building chip GeoDataFrame…"):
            chip_gdf = build_chip_geodataframe(positions, ds, chip_size)

        model, device = _load_resnet()
        prog = st.progress(0.0, text="🧠 Embedding chips…")

        def _cb(done, total):
            prog.progress(done / total, text=f"🧠 Embedding {done}/{total}…")

        embs = embed_chips(chips, model, device, progress_callback=_cb)
        prog.empty()

        faiss_idx = build_index(embs)

        save_chips(chips, chips_path)
        save_embeddings(embs, embs_path)
        save_index(faiss_idx, idx_path)
        save_meta({"positions": positions, "chip_gdf": chip_gdf}, meta_path)

        meta = {"positions": positions, "chip_gdf": chip_gdf}

    st.session_state.chips      = chips
    st.session_state.positions  = meta["positions"]
    st.session_state.embeddings = embs
    st.session_state.faiss_idx  = faiss_idx
    st.session_state.chip_gdf   = meta["chip_gdf"]
    st.session_state.scene_id   = scene_id

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH FLOW — find visually similar chips using a description
# ══════════════════════════════════════════════════════════════════════════════
def run_search(user_query: str, item) -> str:
    stride = int(chip_size * 0.5)
    ensure_index(item, stride)

    chips     = st.session_state.chips
    embs      = st.session_state.embeddings
    faiss_idx = st.session_state.faiss_idx
    chip_gdf  = st.session_state.chip_gdf
    n_chips   = len(chips)

    # Ask the VLM: "which chip index best represents '<query>'?"
    # Build a 4x4 sample mosaic of chips with their indices labelled,
    # then ask the model to pick the best starting chip.
    sample_n   = min(16, n_chips)
    step       = max(1, n_chips // sample_n)
    sample_ids = list(range(0, n_chips, step))[:sample_n]

    # Build a labelled mosaic image
    cols_per_row = 4
    rows_count   = (len(sample_ids) + cols_per_row - 1) // cols_per_row
    mosaic_w     = chip_size * cols_per_row
    mosaic_h     = chip_size * rows_count + 24 * rows_count  # room for labels
    mosaic       = Image.new("RGB", (mosaic_w, mosaic_h + 24), (20, 20, 30))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mosaic)

    for j, sid in enumerate(sample_ids):
        row_j = j // cols_per_row
        col_j = j % cols_per_row
        rgb   = (chips[sid].transpose(1, 2, 0) * 255).astype(np.uint8)
        chip_img = Image.fromarray(rgb)
        x = col_j * chip_size
        y = row_j * (chip_size + 24)
        mosaic.paste(chip_img, (x, y))
        draw.text((x + 4, y + chip_size + 4), f"#{sid}", fill=(150, 200, 255))

    mosaic_io = io.BytesIO()
    mosaic.save(mosaic_io, format="PNG")
    mosaic_io.seek(0)
    mosaic_b64 = base64.b64encode(mosaic_io.read()).decode("utf-8")

    client = get_client()
    pick_prompt = (
        f"The user wants to find: \"{user_query}\".\n"
        f"Each tile is labelled with its chip index (e.g. #42). "
        f"Which single chip index best matches what they are looking for? "
        f"Reply with ONLY the number, no text."
    )

    query_idx = None
    if client:
        try:
            resp = client.chat.completions.create(
                model=ollama_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": pick_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{mosaic_b64}"
                        }},
                    ]
                }],
                max_tokens=10,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip().replace("#", "")
            nums = [int(x) for x in raw.split() if x.isdigit()]
            if nums and 0 <= nums[0] < n_chips:
                query_idx = nums[0]
        except Exception as e:
            log.warning(f"VLM chip selection failed: {e}")

    if query_idx is None:
        # Fallback: pick the middle chip
        query_idx = n_chips // 2

    result_indices, result_scores = query_index(faiss_idx, embs, query_idx, top_k)

    # Store for display
    st.session_state.result_chips  = [chips[i] for i in result_indices]
    st.session_state.result_scores = result_scores
    st.session_state.result_indices = result_indices
    st.session_state.query_chip_idx = query_idx

    # Build result map
    center = (config.DEFAULT_LAT, config.DEFAULT_LON)
    if chip_gdf is not None and len(chip_gdf):
        centroid = chip_gdf.dissolve().centroid.iloc[0]
        center   = (centroid.y, centroid.x)
    fmap = build_result_map(query_idx, result_indices, result_scores, chip_gdf, center)
    # Serialize to HTML for display
    st.session_state.result_map_html = fmap._repr_html_()

    # Ask VLM to narrate what was found
    narration = ""
    if client:
        try:
            # Build a small strip of result chips
            strip_w  = chip_size * min(4, len(result_indices))
            strip    = Image.new("RGB", (strip_w, chip_size), (20, 20, 30))
            for k, idx in enumerate(result_indices[:4]):
                rgb = (chips[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                strip.paste(Image.fromarray(rgb), (k * chip_size, 0))
            strip_io = io.BytesIO()
            strip.save(strip_io, format="PNG")
            strip_io.seek(0)
            strip_b64 = base64.b64encode(strip_io.read()).decode("utf-8")

            narr_resp = client.chat.completions.create(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": config.SEARCH_DESCRIPTION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"The user searched for: \"{user_query}\". Here are the top matching image chips from the aerial scene. Describe what was found in 2-3 sentences."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{strip_b64}"}},
                    ]},
                ],
                max_tokens=200,
            )
            narration = narr_resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"VLM narration failed: {e}")
            narration = f"Found {len(result_indices)} locations matching your query. See the map for their positions."

    return narration

# ══════════════════════════════════════════════════════════════════════════════
# CHAT FLOW — conversational VLM analysis
# ══════════════════════════════════════════════════════════════════════════════
def run_chat(user_text: str) -> str:
    client = get_client()
    if not client:
        return "⚠️ Ollama credentials not set — add them in the sidebar."
    if not st.session_state.naip_b64:
        return "No image loaded yet. Fetch a NAIP tile first using the controls above."

    openai_msgs = [{"role": "system", "content": config.ANALYSIS_SYSTEM_PROMPT}]
    for i, m in enumerate(st.session_state.messages):
        if m["role"] == "assistant":
            openai_msgs.append({"role": "assistant", "content": m["content"]})
        else:
            if i == 0:
                openai_msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": m["content"]},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{st.session_state.naip_b64}"
                        }},
                    ]
                })
            else:
                openai_msgs.append({"role": "user", "content": m["content"]})

    full = ""
    box  = st.empty()
    try:
        stream = client.chat.completions.create(
            model=ollama_model, messages=openai_msgs, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full += delta
            box.markdown(full + "▌")
        box.markdown(full)
    except Exception as e:
        full = f"⚠️ Model error: {e}"
        box.markdown(full)
    return full

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — image left, chat right
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

left, right = st.columns([1, 1], gap="large")

# ── LEFT: image + controls ────────────────────────────────────────────────────
with left:
    st.markdown("#### 📍 Location")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        lat = st.number_input("Lat", value=config.DEFAULT_LAT, format="%.4f")
    with c2:
        lon = st.number_input("Lon", value=config.DEFAULT_LON, format="%.4f")
    with c3:
        buf = st.slider("Buffer °", 0.001, 0.01, 0.003, step=0.001, label_visibility="visible")

    fetch_btn = st.button("🛰️ Load NAIP Tile", use_container_width=True)

    if fetch_btn:
        with st.spinner("Fetching NAIP from Planetary Computer…"):
            try:
                img, b64, item = fetch_naip_point(lat, lon, buf)
                st.session_state.naip_img   = img
                st.session_state.naip_b64   = b64
                st.session_state.naip_scene = item
                st.session_state.messages   = []
                # Reset search results
                st.session_state.result_map_html  = None
                st.session_state.result_chips     = None
                st.session_state.result_scores    = None
                st.success(f"Loaded — {img.width}×{img.height} px  ·  {item.datetime.date()}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("#### 🗺️ NAIP Imagery")
    if st.session_state.naip_img:
        st.image(st.session_state.naip_img, use_container_width=True)

        # Show search result map below the image when available
        if st.session_state.result_map_html:
            st.markdown("#### 📌 Matching Locations")
            st.components.v1.html(st.session_state.result_map_html, height=340)

            if st.session_state.result_chips:
                st.markdown("**Top matches:**")
                n_show  = min(len(st.session_state.result_chips), 8)
                rcols   = st.columns(n_show)
                for k in range(n_show):
                    rgb = (st.session_state.result_chips[k].transpose(1, 2, 0) * 255).astype(np.uint8)
                    score = st.session_state.result_scores[k]
                    rcols[k].image(rgb, use_container_width=True)
                    rcols[k].markdown(
                        f'<p class="result-chip">{score:.3f}</p>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Enter coordinates above and click **Load NAIP Tile** to begin.")

# ── RIGHT: unified chat ───────────────────────────────────────────────────────
with right:
    st.markdown("#### 💬 Ask anything about this image")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Intent pill
            if msg["role"] == "user" and msg.get("intent"):
                intent = msg["intent"]
                cls    = "pill-search" if intent == "SEARCH" else "pill-chat"
                label  = "🔍 SEARCH" if intent == "SEARCH" else "💬 CHAT"
                st.markdown(
                    f'<span class="intent-pill {cls}">{label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

    # Chat input
    placeholder = (
        "Ask a question or describe what to find…"
        if st.session_state.naip_img
        else "Load a NAIP tile first, then ask anything…"
    )
    prompt = st.chat_input(
        placeholder,
        disabled=(not st.session_state.naip_img),
    )

    if prompt:
        # Classify intent
        with st.spinner("Routing…"):
            intent = classify_intent(prompt)

        # Append user message with intent tag
        st.session_state.messages.append({"role": "user", "content": prompt, "intent": intent})

        with st.chat_message("user"):
            cls   = "pill-search" if intent == "SEARCH" else "pill-chat"
            label = "🔍 SEARCH" if intent == "SEARCH" else "💬 CHAT"
            st.markdown(
                f'<span class="intent-pill {cls}">{label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if intent == "SEARCH":
                item = st.session_state.naip_scene
                if item is None:
                    response = "No scene loaded. Fetch a NAIP tile first."
                    st.markdown(response)
                else:
                    with st.spinner(f"🔍 Searching the scene for "{prompt}"…"):
                        response = run_search(prompt, item)
                    st.markdown(response)
                    st.markdown("_Matching locations are shown on the map →_")
            else:
                response = run_chat(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
