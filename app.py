"""
NAIP Intelligence Platform
===========================
One chat interface, three modes routed automatically by intent:

  CHAT      Ask anything about the imagery (VLM analysis)
  SEARCH    Find features visually  (ResNet-50 + FAISS cosine)
  SPECTRAL  Query by spectral signature (NDVI/NDWI/EVI/SAVI/NDBI/Brightness)

AOI selection: draw a rectangle on the interactive map OR enter lat/lon.
Data: USDA NAIP via Microsoft Planetary Computer
"""

import io, logging, base64
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from openai import OpenAI
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image, ImageDraw

import naip_config as config
from utils.imagery import (
    load_naip_scene, chip_scene, chip_scene_4band,
    build_chip_geodataframe, cache_path,
    save_chips, load_chips, save_meta, load_meta,
)
from utils.embeddings import (
    load_model, embed_chips, build_index, query_index, query_index_vec,
    save_index, load_index, save_embeddings, load_embeddings,
)
from utils.spectral import (
    compute_spectral_embeddings, build_spectral_index,
    query_spectral_index, query_spectral_by_chip,
    concept_to_spectral_vector, get_chip_spectral_report,
    save_spectral_index, load_spectral_index,
    save_spectral_embeddings, load_spectral_embeddings,
)
from utils.viz import build_draw_map, build_result_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("naip_platform")

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE, page_icon=config.APP_ICON,
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
  h1,h2,h3                 { font-family:'Space Mono',monospace; }
  .stApp                   { background:#0d1117; color:#c9d1d9; }
  div[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #21262d; }
  .badge { display:inline-block; background:#1f6feb22; color:#58a6ff;
    border:1px solid #1f6feb55; border-radius:4px; padding:2px 10px;
    font-size:0.72rem; font-family:'Space Mono',monospace;
    letter-spacing:0.08em; margin-right:4px; }
  .intent-pill { display:inline-block; border-radius:12px; padding:2px 12px;
    font-size:0.72rem; font-family:'Space Mono',monospace; font-weight:700;
    letter-spacing:0.1em; margin-bottom:6px; }
  .pill-chat     { background:#1f6feb33; color:#58a6ff; border:1px solid #1f6feb66; }
  .pill-search   { background:#23863633; color:#3fb950; border:1px solid #23863666; }
  .pill-spectral { background:#6e40c933; color:#bc8cff; border:1px solid #6e40c966; }
  .result-chip   { text-align:center; font-size:0.7rem; color:#8b949e;
                   font-family:'Space Mono',monospace; }
  .spectral-card { background:#161b22; border:1px solid #21262d; border-radius:8px;
                   padding:10px 14px; margin-bottom:6px; }
  .idx-row  { display:flex; justify-content:space-between; font-size:0.78rem; margin:3px 0; }
  .idx-name { color:#8b949e; font-family:'Space Mono',monospace; }
  .idx-val  { color:#e6edf3; font-weight:600; }
  .idx-bar-wrap { height:6px; background:#21262d; border-radius:3px; margin:2px 0 6px 0; }
  .idx-bar  { height:6px; border-radius:3px; }
  .stButton > button { background:#21262d; color:#c9d1d9; border:1px solid #30363d;
    border-radius:6px; font-family:'Space Mono',monospace; font-size:0.8rem; }
  .stButton > button:hover { background:#238636; color:#fff; border-color:#238636; }
  .hint-box { background:#161b22; border:1px solid #21262d; border-radius:8px;
    padding:10px 16px; font-size:0.78rem; color:#8b949e;
    margin-bottom:12px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🛰️ NAIP Intelligence Platform")
st.markdown(
    '<span class="badge">PLANETARY COMPUTER</span>'
    '<span class="badge">QWEN3-VL</span>'
    '<span class="badge">RESNET-50 · FAISS</span>'
    '<span class="badge">NDVI · NDWI · EVI · SAVI · NDBI</span>',
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "naip_img": None, "naip_b64": None, "naip_scene": None,
    "messages": [],
    "chips": None, "embeddings": None, "faiss_idx": None,
    "chip_gdf": None, "scene_id": None,
    "chips4": None, "spectral_embs": None, "spectral_idx": None,
    "result_map_html": None, "result_chips": None,
    "result_scores": None, "result_indices": None,
    "result_spectral_reports": None,
    "aoi_bbox": None,
    "map_center": [config.DEFAULT_LAT, config.DEFAULT_LON],
    "map_zoom": config.DEFAULT_ZOOM,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    def _s(k, d=""):
        try:    return st.secrets[k]
        except: return d
    ollama_host  = st.text_input("Ollama Host",  value=_s("OLLAMA_HOST",  config.OLLAMA_HOST_DEFAULT))
    ollama_key   = st.text_input("API Key",       value=_s("OLLAMA_API_KEY", config.OLLAMA_KEY_DEFAULT), type="password")
    ollama_model = st.text_input("Model",         value=_s("OLLAMA_MODEL",  config.OLLAMA_MODEL_DEFAULT))
    st.markdown("---")
    st.markdown("**Scene settings**")
    year      = st.selectbox("NAIP Year",       config.NAIP_YEARS, index=2)
    chip_size = st.select_slider("Chip size (px)", [112, 224, 336], value=224)
    top_k     = st.slider("Results to return", 3, 20, config.DEFAULT_TOP_K)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""<div class="hint-box">
<b>Draw:</b> Use the map tab to draw a rectangle over your AOI.<br>
<b>Point:</b> Enter lat/lon in the Lat/Lon tab.<br><br>
<b>Chat:</b> "What land cover is present?"<br>
<b>Visual:</b> "Find all parking lots"<br>
<b>Spectral:</b> "Where is NDVI highest?" · "Find stressed vegetation"
</div>""", unsafe_allow_html=True)
    st.caption("USDA NAIP via Planetary Computer")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client():
    if not ollama_host or not ollama_key: return None
    return OpenAI(base_url=f"{ollama_host.rstrip('/')}/v1", api_key=ollama_key)

def classify_intent(text):
    client = get_client()
    if not client: return "CHAT"
    try:
        r = client.chat.completions.create(
            model=ollama_model,
            messages=[{"role":"system","content":config.INTENT_SYSTEM_PROMPT},
                      {"role":"user","content":text}],
            max_tokens=5, temperature=0)
        w = r.choices[0].message.content.strip().upper()
        if "SPECTRAL" in w: return "SPECTRAL"
        if "SEARCH"   in w: return "SEARCH"
        return "CHAT"
    except: return "CHAT"

def spectral_bar(name, value):
    pct = max(0, min(100, int((value + 1) / 2 * 100)))
    color = ("#3fb950" if value > 0.3 else
             "#58a6ff" if value > 0.0 else
             "#d29922" if value > -0.2 else "#f85149")
    return (f'<div class="idx-row"><span class="idx-name">{name}</span>'
            f'<span class="idx-val">{value:+.3f}</span></div>'
            f'<div class="idx-bar-wrap"><div class="idx-bar" '
            f'style="width:{pct}%;background:{color};"></div></div>')

def _reset_results():
    st.session_state.result_map_html        = None
    st.session_state.result_chips           = None
    st.session_state.result_scores          = None
    st.session_state.result_spectral_reports = None

# ── NAIP fetch ────────────────────────────────────────────────────────────────
def _do_fetch(img, b64, item):
    st.session_state.naip_img   = img
    st.session_state.naip_b64   = b64
    st.session_state.naip_scene = item
    st.session_state.messages   = []
    _reset_results()

def fetch_naip_point(lat, lon, buf):
    catalog = pystac_client.Client.open(config.PC_STAC_URL, modifier=pc.sign_inplace)
    bbox = [lon-buf, lat-buf, lon+buf, lat+buf]
    items = list(catalog.search(
        collections=[config.NAIP_COLLECTION], bbox=bbox,
        limit=1, sortby="-properties.datetime").items())
    if not items: raise ValueError("No NAIP scenes found here.")
    item = items[0]
    with rasterio.open(item.assets["image"].href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        data   = src.read([1,2,3], window=from_bounds(*bounds, transform=src.transform))
    data = np.clip(np.moveaxis(data,0,-1),0,255).astype(np.uint8)
    img  = Image.fromarray(data)
    bio  = io.BytesIO(); img.save(bio,format="PNG"); bio.seek(0)
    return img, base64.b64encode(bio.read()).decode(), item

def fetch_naip_bbox(bbox):
    catalog = pystac_client.Client.open(config.PC_STAC_URL, modifier=pc.sign_inplace)
    items = list(catalog.search(
        collections=[config.NAIP_COLLECTION], bbox=bbox,
        limit=1, sortby="-properties.datetime").items())
    if not items: raise ValueError("No NAIP scenes found in this area.")
    item = items[0]
    with rasterio.open(item.assets["image"].href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        data   = src.read([1,2,3], window=from_bounds(*bounds, transform=src.transform))
    data = np.clip(np.moveaxis(data,0,-1),0,255).astype(np.uint8)
    img  = Image.fromarray(data)
    bio  = io.BytesIO(); img.save(bio,format="PNG"); bio.seek(0)
    return img, base64.b64encode(bio.read()).decode(), item

# ── Index builder ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_resnet(): return load_model()

def ensure_indexes(item):
    scene_id = item.id
    stride   = int(chip_size * 0.5)
    if (st.session_state.scene_id == scene_id
            and st.session_state.faiss_idx is not None
            and st.session_state.spectral_idx is not None):
        return

    cp   = cache_path(scene_id, chip_size, stride, "_chips.npy")
    ep   = cache_path(scene_id, chip_size, stride, "_embeddings.npy")
    ip   = cache_path(scene_id, chip_size, stride, "_faiss.index")
    mp   = cache_path(scene_id, chip_size, stride, "_meta.pkl")
    cp4  = cache_path(scene_id, chip_size, stride, "_chips4.npy")
    sep  = cache_path(scene_id, chip_size, stride, "_spectral_embs.npy")
    sip  = cache_path(scene_id, chip_size, stride, "_spectral.index")

    vc = all(p.exists() for p in [cp, ep, ip, mp])
    sc = all(p.exists() for p in [cp4, sep, sip])

    if vc and sc:
        st.info("Loading indexes from cache...")
        chips=load_chips(cp); embs=load_embeddings(ep)
        fidx=load_index(ip);  meta=load_meta(mp)
        chips4=load_chips(cp4); sembs=load_spectral_embeddings(sep)
        sidx=load_spectral_index(sip)
    else:
        with st.spinner("Loading NAIP scene..."):
            ds, ov = load_naip_scene(item)
        st.success(f"Scene loaded — {ds.shape[1]}x{ds.shape[2]} px (overview {ov})")

        if not vc:
            with st.spinner("Chipping scene (RGB)..."):
                chips, positions = chip_scene(ds, chip_size=chip_size, stride=stride)
                chip_gdf = build_chip_geodataframe(positions, ds, chip_size)
            model, device = _load_resnet()
            prog = st.progress(0.0, text="Visual embeddings...")
            def vcb(done, total): prog.progress(done/total, text=f"Visual {done}/{total}")
            embs = embed_chips(chips, model, device, progress_callback=vcb)
            prog.empty()
            fidx = build_index(embs)
            save_chips(chips, cp); save_embeddings(embs, ep)
            save_index(fidx, ip); save_meta({"positions":positions,"chip_gdf":chip_gdf}, mp)
        else:
            chips=load_chips(cp); embs=load_embeddings(ep); fidx=load_index(ip)

        meta = load_meta(mp)

        if not sc:
            with st.spinner("Chipping scene (RGBI 4-band)..."):
                chips4, _ = chip_scene_4band(ds, chip_size=chip_size, stride=stride)
            prog2 = st.progress(0.0, text="Spectral embeddings...")
            def scb(done, total): prog2.progress(done/total, text=f"Spectral {done}/{total}")
            sembs = compute_spectral_embeddings(chips4, progress_callback=scb)
            prog2.empty()
            sidx  = build_spectral_index(sembs)
            save_chips(chips4, cp4)
            save_spectral_embeddings(sembs, sep); save_spectral_index(sidx, sip)
        else:
            chips4=load_chips(cp4); sembs=load_spectral_embeddings(sep)
            sidx=load_spectral_index(sip)

    st.session_state.chips=chips; st.session_state.embeddings=embs
    st.session_state.faiss_idx=fidx; st.session_state.chip_gdf=meta["chip_gdf"]
    st.session_state.chips4=chips4; st.session_state.spectral_embs=sembs
    st.session_state.spectral_idx=sidx; st.session_state.scene_id=scene_id
    st.success(f"Indexes ready — {len(chips)} chips · visual 2048-d · spectral {sembs.shape[1]}-d")

# ── Shared result store ───────────────────────────────────────────────────────
def _store_results(idxs, scores, chip_gdf, chips, spec_reports=None):
    st.session_state.result_indices=idxs; st.session_state.result_scores=scores
    st.session_state.result_chips=[chips[i] for i in idxs]
    st.session_state.result_spectral_reports=spec_reports
    # Centroid for map center — reproject to avoid geographic CRS warning
    center = (config.DEFAULT_LAT, config.DEFAULT_LON)
    if chip_gdf is not None and len(chip_gdf):
        gdf_proj = chip_gdf.to_crs("EPSG:3857")
        c = gdf_proj.dissolve().centroid.iloc[0]
        c_wgs = gdf_proj.dissolve().to_crs("EPSG:4326").centroid.iloc[0]
        center = (c_wgs.y, c_wgs.x)
    fmap = build_result_map(idxs[0] if idxs else 0, idxs, scores, chip_gdf, center)
    st.session_state.result_map_html = fmap._repr_html_()

# ── Visual search ─────────────────────────────────────────────────────────────
def run_visual_search(query, item):
    ensure_indexes(item)
    chips=st.session_state.chips; embs=st.session_state.embeddings
    fidx=st.session_state.faiss_idx; chip_gdf=st.session_state.chip_gdf
    n=len(chips)

    # Build labelled mosaic for VLM chip selection
    sample_ids=list(range(0, n, max(1, n//16)))[:16]
    cols_n=4; rows_n=(len(sample_ids)+cols_n-1)//cols_n
    mosaic=Image.new("RGB",(chip_size*cols_n,(chip_size+24)*rows_n),(20,20,30))
    draw=ImageDraw.Draw(mosaic)
    for j,sid in enumerate(sample_ids):
        rgb=(chips[sid].transpose(1,2,0)*255).astype(np.uint8)
        r,c=j//cols_n,j%cols_n
        mosaic.paste(Image.fromarray(rgb),(c*chip_size,r*(chip_size+24)))
        draw.text((c*chip_size+4,r*(chip_size+24)+chip_size+4),f"#{sid}",fill=(150,200,255))
    bio=io.BytesIO(); mosaic.save(bio,format="PNG"); bio.seek(0)
    mb64=base64.b64encode(bio.read()).decode()

    qidx=n//2
    client=get_client()
    if client:
        try:
            r=client.chat.completions.create(
                model=ollama_model, max_tokens=10, temperature=0,
                messages=[{"role":"user","content":[
                    {"type":"text","text":(
                        f'User wants to find: "{query}". '
                        f'Each tile is labelled with its chip index (e.g. #42). '
                        f'Which single chip index best matches? Reply with ONLY the number.')},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{mb64}"}},
                ]}])
            raw=r.choices[0].message.content.strip().replace("#","")
            nums=[int(x) for x in raw.split() if x.isdigit()]
            if nums and 0<=nums[0]<n: qidx=nums[0]
        except Exception as e: log.warning(f"VLM chip pick: {e}")

    idxs, scores = query_index(fidx, embs, qidx, top_k)
    _store_results(idxs, scores, chip_gdf, chips)

    narration=f"Found {len(idxs)} visually similar locations."
    if client:
        try:
            strip_n=min(4,len(idxs))
            strip=Image.new("RGB",(chip_size*strip_n,chip_size),(20,20,30))
            for k,idx in enumerate(idxs[:strip_n]):
                strip.paste(Image.fromarray((chips[idx].transpose(1,2,0)*255).astype(np.uint8)),(k*chip_size,0))
            bio=io.BytesIO(); strip.save(bio,format="PNG"); bio.seek(0)
            sb64=base64.b64encode(bio.read()).decode()
            nr=client.chat.completions.create(
                model=ollama_model, max_tokens=200,
                messages=[
                    {"role":"system","content":config.SEARCH_DESCRIPTION_PROMPT},
                    {"role":"user","content":[
                        {"type":"text","text":f'User searched for: "{query}". Describe top matches in 2-3 sentences.'},
                        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{sb64}"}},
                    ]},
                ])
            narration=nr.choices[0].message.content.strip()
        except Exception as e: log.warning(f"Narration: {e}")
    return narration

# ── Spectral search ───────────────────────────────────────────────────────────
def run_spectral_search(query, item):
    ensure_indexes(item)
    chips4=st.session_state.chips4; sembs=st.session_state.spectral_embs
    sidx=st.session_state.spectral_idx; chips=st.session_state.chips
    chip_gdf=st.session_state.chip_gdf

    qvec=concept_to_spectral_vector(query)
    if qvec is not None:
        idxs, scores = query_spectral_index(sidx, sembs, qvec, top_k)
    else:
        client=get_client(); qvec=None
        if client and st.session_state.naip_b64:
            try:
                r=client.chat.completions.create(
                    model=ollama_model, max_tokens=10, temperature=0,
                    messages=[{"role":"user","content":[
                        {"type":"text","text":(
                            f'Query: "{query}". What single land cover concept best describes this? '
                            f'Reply 1-3 words from: dense vegetation, sparse vegetation, stressed vegetation, '
                            f'open water, impervious surface, bare soil, urban, agricultural, wetland, '
                            f'forest, grassland, parking lot, rooftop, high ndvi, low ndvi, high ndwi, '
                            f'high brightness, low brightness.')},
                        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{st.session_state.naip_b64}"}},
                    ]}])
                concept=r.choices[0].message.content.strip().lower()
                qvec=concept_to_spectral_vector(concept)
            except Exception as e: log.warning(f"Concept fallback: {e}")
        if qvec is not None:
            idxs, scores = query_spectral_index(sidx, sembs, qvec, top_k)
        else:
            idxs, scores = query_spectral_by_chip(sidx, sembs, len(chips4)//2, top_k)

    spec_reports=[get_chip_spectral_report(chips4[i]) for i in idxs]
    _store_results(idxs, scores, chip_gdf, chips, spec_reports=spec_reports)

    report_text="\n".join([
        f"Chip {i}: NDVI={r['ndvi']:+.3f} ({r['ndvi_class']}), "
        f"NDWI={r['ndwi']:+.3f} ({r['ndwi_class']}), "
        f"EVI={r['evi']:+.3f}, Brightness={r['brightness']:.3f}"
        for i,r in zip(idxs,spec_reports)])

    narration=f"Found {len(idxs)} spectral matches."
    client=get_client()
    if client:
        try:
            nr=client.chat.completions.create(
                model=ollama_model, max_tokens=250,
                messages=[
                    {"role":"system","content":config.SPECTRAL_DESCRIPTION_PROMPT},
                    {"role":"user","content":f'Query: "{query}"\n\nTop matches:\n{report_text}\n\nDescribe in 2-3 sentences.'},
                ])
            narration=nr.choices[0].message.content.strip()
        except Exception as e: log.warning(f"Spectral narration: {e}")
    return narration

# ── Chat ──────────────────────────────────────────────────────────────────────
def run_chat(text):
    client=get_client()
    if not client: return "Add Ollama credentials in the sidebar."
    if not st.session_state.naip_b64: return "Load a NAIP tile first."
    msgs=[{"role":"system","content":config.ANALYSIS_SYSTEM_PROMPT}]
    for i,m in enumerate(st.session_state.messages):
        if m["role"]=="assistant":
            msgs.append({"role":"assistant","content":m["content"]})
        elif i==0:
            msgs.append({"role":"user","content":[
                {"type":"text","text":m["content"]},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{st.session_state.naip_b64}"}},
            ]})
        else:
            msgs.append({"role":"user","content":m["content"]})
    full=""; box=st.empty()
    try:
        stream=client.chat.completions.create(model=ollama_model,messages=msgs,stream=True)
        for chunk in stream:
            delta=chunk.choices[0].delta.content or ""
            full+=delta; box.markdown(full+"▌")
        box.markdown(full)
    except Exception as e: full=f"Model error: {e}"; box.markdown(full)
    return full

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
left, right = st.columns([1, 1], gap="large")

with left:
    # ── AOI selection tabs ────────────────────────────────────────────────────
    map_tab, latlon_tab = st.tabs(["🗺️ Draw on Map", "📍 Lat / Lon"])

    with map_tab:
        st.caption("Draw a rectangle to define your AOI, then click Load.")
        draw_map = build_draw_map(
            center=tuple(st.session_state.map_center),
            zoom=st.session_state.map_zoom,
            existing_bbox=st.session_state.aoi_bbox,
        )
        map_data = st_folium(
            draw_map, width="100%", height=340,
            returned_objects=["all_drawings"],
            key="aoi_draw_map",
        )

        def _parse_bbox(md):
            try:
                drawings = md.get("all_drawings") or []
                if not drawings: return None
                geom = drawings[-1]["geometry"]
                if geom["type"] != "Polygon": return None
                coords = geom["coordinates"][0]
                lngs=[c[0] for c in coords]; lats=[c[1] for c in coords]
                return [min(lngs), min(lats), max(lngs), max(lats)]
            except Exception: return None

        drawn = _parse_bbox(map_data)
        if drawn and drawn != st.session_state.aoi_bbox:
            w,s,e,n = drawn
            if (e-w) > 0 and (n-s) > 0:
                st.session_state.aoi_bbox = drawn

        if st.session_state.aoi_bbox:
            w,s,e,n = st.session_state.aoi_bbox
            st.caption(f"AOI: W {w:.4f}  S {s:.4f}  E {e:.4f}  N {n:.4f}")
            if (e-w)*(n-s) > 0.25:
                st.warning("Large AOI — draw a smaller box for faster loading.")

        if st.button("🛰️ Load NAIP from AOI",
                     disabled=(st.session_state.aoi_bbox is None),
                     use_container_width=True, key="map_load_btn"):
            with st.spinner("Fetching NAIP..."):
                try:
                    img,b64,item = fetch_naip_bbox(st.session_state.aoi_bbox)
                    _do_fetch(img, b64, item)
                    w,s,e,n = st.session_state.aoi_bbox
                    st.session_state.map_center = [(s+n)/2, (w+e)/2]
                    st.success(f"Loaded — {img.width}x{img.height} px · {item.datetime.date()}")
                except Exception as e: st.error(str(e))

    with latlon_tab:
        st.caption("Enter a point coordinate and buffer to define the area.")
        c1,c2,c3 = st.columns([2,2,1])
        with c1: lat = st.number_input("Lat", value=config.DEFAULT_LAT, format="%.4f")
        with c2: lon = st.number_input("Lon", value=config.DEFAULT_LON, format="%.4f")
        with c3: buf = st.slider("Buffer", 0.001, 0.01, 0.003, step=0.001)

        if st.button("🛰️ Load NAIP from Point",
                     use_container_width=True, key="ll_load_btn"):
            with st.spinner("Fetching..."):
                try:
                    img,b64,item = fetch_naip_point(lat, lon, buf)
                    _do_fetch(img, b64, item)
                    st.session_state.aoi_bbox   = [lon-buf, lat-buf, lon+buf, lat+buf]
                    st.session_state.map_center = [lat, lon]
                    st.session_state.map_zoom   = 13
                    st.success(f"Loaded — {img.width}x{img.height} px · {item.datetime.date()}")
                except Exception as e: st.error(str(e))

    # ── Imagery display ───────────────────────────────────────────────────────
    st.markdown("#### NAIP Imagery")
    if st.session_state.naip_img:
        st.image(st.session_state.naip_img, use_container_width=True)

        if st.session_state.result_map_html:
            st.markdown("#### Matching Locations")
            st.iframe(st.session_state.result_map_html, height=300)

            rc=st.session_state.result_chips or []
            rs=st.session_state.result_scores or []
            sr=st.session_state.result_spectral_reports
            n_show=min(len(rc),8)
            if n_show:
                st.markdown("**Top matches:**")
                rcols=st.columns(n_show)
                for k in range(n_show):
                    rgb=(rc[k].transpose(1,2,0)*255).astype(np.uint8)
                    rcols[k].image(rgb, use_container_width=True)
                    rcols[k].markdown(
                        f'<p class="result-chip">{rs[k]:.3f}</p>',
                        unsafe_allow_html=True)

            if sr:
                st.markdown("#### Spectral Index Summary")
                for k,r in enumerate(sr[:4]):
                    bars=(spectral_bar("NDVI",r["ndvi"])+spectral_bar("NDWI",r["ndwi"])+
                          spectral_bar("EVI",r["evi"])+spectral_bar("Brightness",r["brightness"]*2-1))
                    st.markdown(
                        f'<div class="spectral-card">'
                        f'<div style="font-size:0.72rem;color:#8b949e;font-family:Space Mono,monospace;margin-bottom:6px;">'
                        f'Match #{k+1} · {r["ndvi_class"]} · {r["ndwi_class"]}</div>'
                        f'{bars}</div>', unsafe_allow_html=True)
    else:
        st.info("Draw an AOI on the map or enter coordinates, then click Load.")

# ── Right: chat ───────────────────────────────────────────────────────────────
with right:
    st.markdown("#### Ask anything about this image")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"]=="user" and msg.get("intent"):
                i=msg["intent"]
                cls,label=(("pill-spectral","SPECTRAL") if i=="SPECTRAL" else
                            ("pill-search","SEARCH") if i=="SEARCH" else ("pill-chat","CHAT"))
                st.markdown(f'<span class="intent-pill {cls}">{label}</span>',
                            unsafe_allow_html=True)
            st.markdown(msg["content"])

    prompt=st.chat_input(
        "Ask a question, or describe what to find visually or spectrally...",
        disabled=(not st.session_state.naip_img))

    if prompt:
        with st.spinner("Routing..."):
            intent=classify_intent(prompt)
        st.session_state.messages.append({"role":"user","content":prompt,"intent":intent})

        with st.chat_message("user"):
            cls,label=(("pill-spectral","SPECTRAL") if intent=="SPECTRAL" else
                        ("pill-search","SEARCH") if intent=="SEARCH" else ("pill-chat","CHAT"))
            st.markdown(f'<span class="intent-pill {cls}">{label}</span>',unsafe_allow_html=True)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            item=st.session_state.naip_scene
            if intent=="SPECTRAL":
                if not item: response="Load a NAIP tile first."; st.markdown(response)
                else:
                    with st.spinner("Running spectral search..."):
                        response=run_spectral_search(prompt,item)
                    st.markdown(response)
                    st.markdown("_Spectral matches and index cards shown on the left_")
            elif intent=="SEARCH":
                if not item: response="Load a NAIP tile first."; st.markdown(response)
                else:
                    with st.spinner("Searching visually..."):
                        response=run_visual_search(prompt,item)
                    st.markdown(response)
                    st.markdown("_Matching locations shown on the map_")
            else:
                response=run_chat(prompt)

        st.session_state.messages.append({"role":"assistant","content":response})
        st.rerun()
