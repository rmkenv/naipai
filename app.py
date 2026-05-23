"""
NAIP Intelligence Platform — single-file, no utils imports that can break
"""
import io, logging, base64, hashlib, pickle
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
import folium
import folium.plugins
import geopandas as gpd
from shapely.geometry import box, mapping
import faiss
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("naip")

# ── Config (inline — no separate file needed) ─────────────────────────────────
PC_STAC   = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COL  = "naip"
ESRI_URL  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
DEF_LAT, DEF_LON, DEF_ZOOM = 40.7128, -74.0060, 12
NAIP_YEARS = [2023,2022,2021,2020,2019,2018,2017]
CHIP_SIZE  = 224
MAX_CHIPS  = 2000
IMG_MEAN   = [0.485,0.456,0.406]
IMG_STD    = [0.229,0.224,0.225]

CACHE_DIR = Path("/tmp/naip_cache")
CACHE_DIR.mkdir(exist_ok=True)

try:
    import naip_config as _nc
    OLLAMA_HOST_DEF  = _nc.OLLAMA_HOST_DEFAULT
    OLLAMA_KEY_DEF   = _nc.OLLAMA_KEY_DEFAULT
    OLLAMA_MODEL_DEF = _nc.OLLAMA_MODEL_DEFAULT
    INTENT_PROMPT    = _nc.INTENT_SYSTEM_PROMPT
    ANALYSIS_PROMPT  = _nc.ANALYSIS_SYSTEM_PROMPT
    SEARCH_PROMPT    = _nc.SEARCH_DESCRIPTION_PROMPT
    SPECTRAL_PROMPT  = _nc.SPECTRAL_DESCRIPTION_PROMPT
except Exception:
    import os
    OLLAMA_HOST_DEF  = os.getenv("OLLAMA_HOST","")
    OLLAMA_KEY_DEF   = os.getenv("OLLAMA_API_KEY","")
    OLLAMA_MODEL_DEF = os.getenv("OLLAMA_MODEL","qwen3-vl:235b-cloud")
    INTENT_PROMPT = "Classify as CHAT, SEARCH, or SPECTRAL. Reply one word only."
    ANALYSIS_PROMPT = "You are an expert aerial imagery analyst."
    SEARCH_PROMPT   = "Describe the visually similar locations found."
    SPECTRAL_PROMPT = "Describe the spectral signature of these chips."

# ── Spectral archetypes ───────────────────────────────────────────────────────
ARCHETYPES = {
    "dense vegetation":    [0.75,0.05,-0.50,0.05,0.55,0.05,0.55,0.05,-0.50,0.05,0.35,0.05],
    "sparse vegetation":   [0.30,0.10,-0.15,0.08,0.20,0.08,0.22,0.08,-0.10,0.08,0.40,0.08],
    "stressed vegetation": [0.15,0.12, 0.00,0.10,0.08,0.10,0.10,0.10, 0.05,0.10,0.38,0.10],
    "open water":          [-0.15,0.05,0.25,0.06,-0.10,0.05,-0.12,0.05,-0.20,0.05,0.20,0.05],
    "impervious surface":  [-0.10,0.08,-0.10,0.08,0.00,0.08,0.00,0.08,0.30,0.08,0.55,0.10],
    "bare soil":           [0.05,0.08,-0.05,0.08,0.03,0.08,0.04,0.08,0.20,0.08,0.50,0.10],
    "urban":               [-0.05,0.10,-0.08,0.08,0.00,0.10,0.00,0.10,0.35,0.10,0.60,0.12],
    "agricultural":        [0.55,0.10,-0.30,0.08,0.40,0.10,0.40,0.10,-0.30,0.08,0.38,0.08],
    "forest":              [0.80,0.05,-0.55,0.05,0.60,0.05,0.60,0.05,-0.55,0.05,0.30,0.05],
    "grassland":           [0.40,0.12,-0.20,0.08,0.28,0.10,0.30,0.10,-0.18,0.08,0.42,0.08],
    "parking lot":         [-0.08,0.05,-0.08,0.05,0.00,0.05,0.00,0.05,0.38,0.06,0.58,0.08],
    "rooftop":             [-0.05,0.08,-0.06,0.06,0.00,0.08,0.00,0.08,0.32,0.10,0.65,0.12],
    "high ndvi":           [0.75,0.05,-0.50,0.05,0.55,0.05,0.55,0.05,-0.50,0.05,0.35,0.05],
    "high ndwi":           [-0.20,0.05,0.30,0.06,-0.12,0.05,-0.15,0.05,-0.25,0.05,0.18,0.05],
    "high brightness":     [-0.02,0.06,-0.05,0.06,0.00,0.06,0.00,0.06,0.35,0.08,0.70,0.10],
    "wetland":             [0.35,0.12,0.15,0.10,0.25,0.10,0.28,0.10,-0.15,0.08,0.28,0.08],
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NAIP Intelligence", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace;}
.stApp{background:#0d1117;color:#c9d1d9;}
div[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #21262d;}
.badge{display:inline-block;background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb55;
  border-radius:4px;padding:2px 10px;font-size:.72rem;font-family:'Space Mono',monospace;
  letter-spacing:.08em;margin-right:4px;}
.pill{display:inline-block;border-radius:12px;padding:2px 12px;font-size:.72rem;
  font-family:'Space Mono',monospace;font-weight:700;letter-spacing:.1em;margin-bottom:6px;}
.pill-chat    {background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb66;}
.pill-search  {background:#23863633;color:#3fb950;border:1px solid #23863666;}
.pill-spectral{background:#6e40c933;color:#bc8cff;border:1px solid #6e40c966;}
.tile-label{text-align:center;font-size:.75rem;font-family:'Space Mono',monospace;
  padding:3px 0;margin-top:2px;}
.query-label{color:#00e5ff;font-weight:700;}
.result-label{color:#8b949e;}
.stButton>button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;font-family:'Space Mono',monospace;font-size:.8rem;}
.stButton>button:hover{background:#238636;color:#fff;border-color:#238636;}
</style>""", unsafe_allow_html=True)

st.markdown("# 🛰️ NAIP Intelligence Platform")
st.markdown('<span class="badge">PLANETARY COMPUTER</span>'
            '<span class="badge">QWEN3-VL</span>'
            '<span class="badge">RESNET-50·FAISS</span>'
            '<span class="badge">NDVI·NDWI·EVI·SAVI</span>',
            unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
SS = {
    "tile_bytes": None,      # PNG bytes of loaded NAIP tile
    "tile_b64": None,        # base64 for VLM
    "tile_scene": None,      # STAC item
    "messages": [],
    # Index state
    "chips_rgb": None,       # (N,3,H,W) float32
    "chips_4b":  None,       # (N,4,H,W) float32
    "embs_vis":  None,       # (N,2048) L2-normed
    "embs_spec": None,       # (N,12)   L2-normed
    "idx_vis":   None,       # faiss index
    "idx_spec":  None,       # faiss index
    "chip_gdf":  None,       # GeoDataFrame
    "scene_id":  None,
    # Results — stored as PNG bytes lists
    "res_query_bytes": None, # PNG bytes of the query chip
    "res_chip_bytes": [],    # list of PNG bytes
    "res_scores": [],
    "res_map_html": None,
    "res_spec_reports": None,
    "res_label": "",         # description text
    # AOI
    "aoi_bbox": None,
    "map_center": [DEF_LAT, DEF_LON],
    "map_zoom": DEF_ZOOM,
}
for k,v in SS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    def _s(k,d=""):
        try: return st.secrets[k]
        except: return d
    ollama_host  = st.text_input("Ollama Host",  value=_s("OLLAMA_HOST",  OLLAMA_HOST_DEF))
    ollama_key   = st.text_input("API Key",       value=_s("OLLAMA_API_KEY",OLLAMA_KEY_DEF), type="password")
    ollama_model = st.text_input("Model",         value=_s("OLLAMA_MODEL",  OLLAMA_MODEL_DEF))
    st.markdown("---")
    top_k = st.slider("Similar tiles to find", 3, 8, 6)
    year  = st.selectbox("NAIP Year", NAIP_YEARS, index=2)
    st.markdown("---")
    st.markdown("""**Usage:**
- Draw rectangle on map → Load
- Or enter Lat/Lon → Load
- Chat: *"What's the land cover?"*
- Find: *"Find parking lots"*
- Spectral: *"Where is NDVI highest?"*""")

# ── Helpers ───────────────────────────────────────────────────────────────────
def client():
    if not ollama_host or not ollama_key: return None
    return OpenAI(base_url=f"{ollama_host.rstrip('/')}/v1", api_key=ollama_key)

def img_to_bytes(img):
    b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()

def arr_to_bytes(arr_chw_float):
    """(3,H,W) float32 [0,1] → PNG bytes"""
    rgb = (arr_chw_float.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    return img_to_bytes(Image.fromarray(rgb))

def classify_intent(text):
    c = client()
    if not c: return "CHAT"
    try:
        r = c.chat.completions.create(
            model=ollama_model,
            messages=[{"role":"system","content":INTENT_PROMPT},
                      {"role":"user","content":text}],
            max_tokens=5, temperature=0)
        w = r.choices[0].message.content.strip().upper()
        if "SPECTRAL" in w: return "SPECTRAL"
        if "SEARCH"   in w: return "SEARCH"
        return "CHAT"
    except: return "CHAT"

# ── NAIP fetch ────────────────────────────────────────────────────────────────
def fetch_tile(bbox):
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[NAIP_COL], bbox=bbox,
                            limit=1, sortby="-properties.datetime").items())
    if not items: raise ValueError("No NAIP scenes found here.")
    item = items[0]
    with rasterio.open(item.assets["image"].href) as src:
        bnds = transform_bounds("EPSG:4326", src.crs, *bbox)
        data = src.read([1,2,3], window=from_bounds(*bnds, transform=src.transform))
    data = np.clip(np.moveaxis(data,0,-1),0,255).astype(np.uint8)
    img  = Image.fromarray(data)
    bio  = io.BytesIO(); img.save(bio,format="PNG"); bio.seek(0)
    b64  = base64.b64encode(bio.read()).decode()
    return img_to_bytes(img), b64, item

def reset_tile(tile_bytes, b64, item):
    st.session_state.tile_bytes  = tile_bytes
    st.session_state.tile_b64    = b64
    st.session_state.tile_scene  = item
    st.session_state.messages    = []
    st.session_state.res_chip_bytes  = []
    st.session_state.res_scores      = []
    st.session_state.res_map_html    = None
    st.session_state.res_spec_reports= None
    st.session_state.res_query_bytes = None
    st.session_state.res_label       = ""

# ── Chipping ──────────────────────────────────────────────────────────────────
def _norm(arr): # per-image percentile stretch
    lo,hi = np.percentile(arr,2), np.percentile(arr,98)
    return np.clip((arr-lo)/(hi-lo+1e-8),0,1).astype(np.float32)

def chip(ds, n_bands=3, stride=None):
    if stride is None: stride = CHIP_SIZE//2
    arr = _norm(ds.values[:n_bands].astype(np.float32))
    if n_bands==4 and ds.values.shape[0]<4:
        pad = np.zeros((1,arr.shape[1],arr.shape[2]),dtype=np.float32)
        arr = np.concatenate([arr,pad],0)
    _,H,W = arr.shape
    chips,pos=[],[]
    for y in range(0,H-CHIP_SIZE,stride):
        for x in range(0,W-CHIP_SIZE,stride):
            c = arr[:,y:y+CHIP_SIZE,x:x+CHIP_SIZE]
            if c.shape[-2:]==(CHIP_SIZE,CHIP_SIZE):
                chips.append(c); pos.append((y,x))
            if len(chips)>=MAX_CHIPS: break
        if len(chips)>=MAX_CHIPS: break
    return np.stack(chips), pos

# ── Georeferencing ────────────────────────────────────────────────────────────
def make_gdf(positions, ds, crs="EPSG:4326"):
    tf = ds.rio.transform()
    geoms=[]
    for row,col in positions:
        l = tf.c+col*tf.a; t = tf.f+row*tf.e
        geoms.append(box(l, t+CHIP_SIZE*tf.e, l+CHIP_SIZE*tf.a, t))
    gdf = gpd.GeoDataFrame({"chip_id":range(len(geoms))},
                            geometry=geoms, crs=ds.rio.crs or crs)
    return gdf.to_crs(crs) if str(gdf.crs)!=crs else gdf

# ── Embedding ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resnet():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity(); m.eval()
    return m, "cuda" if torch.cuda.is_available() else "cpu"

_tnorm = T.Normalize(mean=IMG_MEAN, std=IMG_STD)

def embed_rgb(chips, prog=None):
    model, dev = load_resnet()
    all_e=[]; total=len(chips); bs=32
    for i in range(0,total,bs):
        b = torch.tensor(chips[i:i+bs], dtype=torch.float32, device=dev)
        b = torch.stack([_tnorm(x) for x in b])
        with torch.no_grad(): all_e.append(model(b).cpu().numpy())
        if prog: prog.progress(min(i+bs,total)/total, text=f"Embedding {min(i+bs,total)}/{total}")
    e = np.concatenate(all_e,0).astype(np.float32)
    faiss.normalize_L2(e); return e

# ── Spectral ──────────────────────────────────────────────────────────────────
def _sr(a,b):
    d=a+b; return np.where(np.abs(d)>1e-6,(a-b)/d,0.0)

def spectral_vec(chip4):
    R,G,B,N=chip4[0],chip4[1],chip4[2],chip4[3]
    ndvi=_sr(N,R); ndwi=_sr(G,N)
    ed=N+6*R-7.5*B+1; evi=np.clip(np.where(np.abs(ed)>1e-6,2.5*(N-R)/ed,0),-2,2)
    sd=N+R+0.5; savi=np.where(np.abs(sd)>1e-6,((N-R)/sd)*1.5,0)
    ndbi=_sr(R,N); br=(R+G+B+N)/4
    f=[]
    for a in [ndvi,ndwi,evi,savi,ndbi,br]: f+=[float(np.mean(a)),float(np.std(a))]
    return np.array(f,dtype=np.float32)

def embed_spectral(chips4, prog=None):
    N=chips4.shape[0]; e=np.zeros((N,12),dtype=np.float32)
    for i in range(N):
        e[i]=spectral_vec(chips4[i])
        if prog and i%50==0: prog.progress(i/N, text=f"Spectral {i}/{N}")
    faiss.normalize_L2(e); return e

def spectral_report(chip4):
    R,G,B,N=chip4[0],chip4[1],chip4[2],chip4[3]
    ndvi=float(np.mean(_sr(N,R))); ndwi=float(np.mean(_sr(G,N)))
    ed=N+6*R-7.5*B+1; evi=float(np.mean(np.where(np.abs(ed)>1e-6,2.5*(N-R)/ed,0)))
    br=float(np.mean((R+G+B+N)/4))
    def nc(v): return ("dense veg" if v>.6 else "mod. veg" if v>.3 else
                       "sparse veg" if v>.1 else "bare/built" if v>0 else "water/imp.")
    def wc(v): return "water" if v>.2 else "moist" if v>0 else "dry"
    return {"ndvi":round(ndvi,3),"ndwi":round(ndwi,3),"evi":round(evi,3),
            "brightness":round(br,3),"nc":nc(ndvi),"wc":wc(ndwi)}

def concept_vec(text):
    lc=text.lower(); best,bscore=None,0
    for k in ARCHETYPES:
        s=sum(1 for w in k.split() if w in lc)
        if s>bscore: bscore=s; best=k
    if not best or bscore==0: return None
    v=np.array(ARCHETYPES[best],dtype=np.float32)
    faiss.normalize_L2(v.reshape(1,-1)); return v

# ── Cache helpers ─────────────────────────────────────────────────────────────
def _ckey(sid,stride): return hashlib.md5(f"{sid}_{CHIP_SIZE}_{stride}".encode()).hexdigest()[:12]
def _cp(sid,stride,sfx): return CACHE_DIR/f"{_ckey(sid,stride)}{sfx}"

# ── Index builder ─────────────────────────────────────────────────────────────
def ensure_index(item):
    import rioxarray
    sid=item.id; stride=CHIP_SIZE//2
    if st.session_state.scene_id==sid and st.session_state.idx_vis is not None:
        return

    p_rgb =_cp(sid,stride,"_rgb.npy");  p_4b=_cp(sid,stride,"_4b.npy")
    p_evis=_cp(sid,stride,"_evis.npy"); p_espc=_cp(sid,stride,"_espc.npy")
    p_iv  =_cp(sid,stride,"_iv.index"); p_is=_cp(sid,stride,"_is.index")
    p_gdf =_cp(sid,stride,"_gdf.pkl")

    if all(p.exists() for p in [p_rgb,p_4b,p_evis,p_espc,p_iv,p_is,p_gdf]):
        st.info("Loading from cache...")
        chips_rgb=np.load(p_rgb); chips_4b=np.load(p_4b)
        evis=np.load(p_evis);     espc=np.load(p_espc)
        iv=faiss.read_index(str(p_iv)); isp=faiss.read_index(str(p_is))
        with open(p_gdf,"rb") as f: gdf=pickle.load(f)
    else:
        with st.spinner("Loading NAIP scene..."):
            ds = rioxarray.open_rasterio(item.assets["image"].href, overview_level=2)

        with st.spinner("Chipping (RGB)..."):
            chips_rgb, pos = chip(ds, n_bands=3)

        with st.spinner("Chipping (RGBI)..."):
            chips_4b, _   = chip(ds, n_bands=4)

        with st.spinner("Building GeoDataFrame..."):
            gdf = make_gdf(pos, ds)

        p1=st.progress(0.0,text="Visual embeddings...")
        evis=embed_rgb(chips_rgb, prog=p1); p1.empty()

        p2=st.progress(0.0,text="Spectral embeddings...")
        espc=embed_spectral(chips_4b, prog=p2); p2.empty()

        iv=faiss.IndexFlatIP(evis.shape[1]); iv.add(evis)
        isp=faiss.IndexFlatIP(espc.shape[1]); isp.add(espc)

        np.save(p_rgb,chips_rgb); np.save(p_4b,chips_4b)
        np.save(p_evis,evis);     np.save(p_espc,espc)
        faiss.write_index(iv,str(p_iv)); faiss.write_index(isp,str(p_is))
        with open(p_gdf,"wb") as f: pickle.dump(gdf,f)

    st.session_state.chips_rgb=chips_rgb; st.session_state.chips_4b=chips_4b
    st.session_state.embs_vis=evis;       st.session_state.embs_spec=espc
    st.session_state.idx_vis=iv;          st.session_state.idx_spec=isp
    st.session_state.chip_gdf=gdf;        st.session_state.scene_id=sid
    st.success(f"Index ready — {len(chips_rgb)} chips")

# ── FAISS query ───────────────────────────────────────────────────────────────
def query_by_idx(index, embs, qidx, k):
    qv=embs[qidx:qidx+1].copy()
    D,I=index.search(qv,k+1)
    idxs,scs=[],[]
    for i,s in zip(I[0],D[0]):
        if i==qidx: continue
        idxs.append(int(i)); scs.append(float(s))
        if len(idxs)==k: break
    return idxs,scs

def query_by_vec(index, qvec, k):
    qv=qvec.reshape(1,-1).astype(np.float32).copy()
    faiss.normalize_L2(qv)
    D,I=index.search(qv,k)
    return [int(i) for i in I[0]],[float(s) for s in D[0]]

# ── Result map ────────────────────────────────────────────────────────────────
def make_result_map(qidx, idxs, scores, gdf, center):
    m=folium.Map(location=list(center),zoom_start=14,tiles=None)
    folium.TileLayer(ESRI_URL,attr=ESRI_ATTR,name="Satellite",max_zoom=19).add_to(m)
    row=gdf.loc[gdf["chip_id"]==qidx,"geometry"]
    if not row.empty:
        folium.GeoJson(mapping(row.iloc[0]),
            style_function=lambda _:{"fillColor":"#00e5ff","color":"#00e5ff",
                                      "weight":2,"fillOpacity":0.4},
            tooltip="QUERY").add_to(m)
    mn,mx=(min(scores),max(scores)) if scores else (0,1)
    for rank,(i,s) in enumerate(zip(idxs,scores),1):
        row=gdf.loc[gdf["chip_id"]==i,"geometry"]
        if row.empty: continue
        norm=(s-mn)/(mx-mn+1e-8); r=int(255*(1-norm))
        color=f"#{r:02x}d23c"
        folium.GeoJson(mapping(row.iloc[0]),
            style_function=lambda _,c=color:{"fillColor":c,"color":"#fff",
                                              "weight":1,"fillOpacity":0.5},
            tooltip=f"#{rank} sim={s:.3f}").add_to(m)
    return m._repr_html_()

def map_center(gdf):
    try:
        c=gdf.to_crs("EPSG:3857").dissolve().to_crs("EPSG:4326").centroid.iloc[0]
        return (c.y,c.x)
    except: return (DEF_LAT,DEF_LON)

# ── Store results ─────────────────────────────────────────────────────────────
def store_results(qidx, idxs, scores, chips_rgb, chips_4b, gdf, spec_reports=None):
    st.session_state.res_query_bytes = arr_to_bytes(chips_rgb[qidx])
    st.session_state.res_chip_bytes  = [arr_to_bytes(chips_rgb[i]) for i in idxs]
    st.session_state.res_scores      = list(scores)
    st.session_state.res_spec_reports= spec_reports
    center = map_center(gdf)
    st.session_state.res_map_html = make_result_map(qidx, idxs, scores, gdf, center)

# ── VLM: pick best query chip from mosaic ─────────────────────────────────────
def pick_query_chip(chips_rgb, query_text):
    n=len(chips_rgb)
    sample=list(range(0,n,max(1,n//16)))[:16]
    nc=4; nr=(len(sample)+nc-1)//nc
    mosaic=Image.new("RGB",(CHIP_SIZE*nc,(CHIP_SIZE+20)*nr),(20,20,30))
    draw=ImageDraw.Draw(mosaic)
    for j,sid in enumerate(sample):
        rgb=(chips_rgb[sid].transpose(1,2,0)*255).astype(np.uint8)
        r,c=j//nc,j%nc
        mosaic.paste(Image.fromarray(rgb),(c*CHIP_SIZE,r*(CHIP_SIZE+20)))
        draw.text((c*CHIP_SIZE+4,r*(CHIP_SIZE+20)+CHIP_SIZE+2),f"#{sid}",fill=(150,200,255))
    bio=io.BytesIO(); mosaic.save(bio,format="PNG"); bio.seek(0)
    mb64=base64.b64encode(bio.read()).decode()
    c=client()
    if c:
        try:
            r=c.chat.completions.create(model=ollama_model,max_tokens=10,temperature=0,
                messages=[{"role":"user","content":[
                    {"type":"text","text":f'Find: "{query_text}". Which chip index matches best? Reply ONLY the number.'},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{mb64}"}},
                ]}])
            raw=r.choices[0].message.content.strip().replace("#","")
            nums=[int(x) for x in raw.split() if x.isdigit()]
            if nums and 0<=nums[0]<n: return nums[0]
        except: pass
    return n//2

# ── VLM: narrate results ──────────────────────────────────────────────────────
def narrate(query, chips_rgb, idxs, sys_prompt):
    c=client()
    if not c: return f"Found {len(idxs)} matches."
    strip_n=min(4,len(idxs))
    strip=Image.new("RGB",(CHIP_SIZE*strip_n,CHIP_SIZE),(20,20,30))
    for k,i in enumerate(idxs[:strip_n]):
        strip.paste(Image.fromarray((chips_rgb[i].transpose(1,2,0)*255).astype(np.uint8)),(k*CHIP_SIZE,0))
    bio=io.BytesIO(); strip.save(bio,format="PNG"); bio.seek(0)
    sb64=base64.b64encode(bio.read()).decode()
    try:
        r=c.chat.completions.create(model=ollama_model,max_tokens=200,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":[
                          {"type":"text","text":f'Query: "{query}". Describe top matches in 2-3 sentences.'},
                          {"type":"image_url","image_url":{"url":f"data:image/png;base64,{sb64}"}},
                      ]}])
        return r.choices[0].message.content.strip()
    except Exception as e: return f"Found {len(idxs)} matches. ({e})"

# ── Search handlers ───────────────────────────────────────────────────────────
def run_visual_search(query, item):
    ensure_index(item)
    chips_rgb=st.session_state.chips_rgb; embs=st.session_state.embs_vis
    iv=st.session_state.idx_vis; gdf=st.session_state.chip_gdf
    chips_4b=st.session_state.chips_4b
    with st.spinner("Selecting query chip..."): qidx=pick_query_chip(chips_rgb,query)
    with st.spinner("Searching..."): idxs,scores=query_by_idx(iv,embs,qidx,top_k)
    store_results(qidx,idxs,scores,chips_rgb,chips_4b,gdf)
    return narrate(query,chips_rgb,idxs,SEARCH_PROMPT)

def run_spectral_search(query, item):
    ensure_index(item)
    chips_rgb=st.session_state.chips_rgb; chips_4b=st.session_state.chips_4b
    espc=st.session_state.embs_spec; isp=st.session_state.idx_spec
    gdf=st.session_state.chip_gdf
    qvec=concept_vec(query)
    if qvec is not None:
        idxs,scores=query_by_vec(isp,qvec,top_k)
        qidx=idxs[0]
    else:
        qidx=len(chips_rgb)//2
        idxs,scores=query_by_idx(isp,espc,qidx,top_k)
    spec_reports=[spectral_report(chips_4b[i]) for i in idxs]
    store_results(qidx,idxs,scores,chips_rgb,chips_4b,gdf,spec_reports=spec_reports)
    rtext="\n".join([f"Chip {i}: NDVI={r['ndvi']:+.3f} ({r['nc']}), NDWI={r['ndwi']:+.3f}"
                     for i,r in zip(idxs,spec_reports)])
    c=client()
    if not c: return f"Found {len(idxs)} spectral matches."
    try:
        r=c.chat.completions.create(model=ollama_model,max_tokens=250,
            messages=[{"role":"system","content":SPECTRAL_PROMPT},
                      {"role":"user","content":f'Query: "{query}"\n{rtext}\nDescribe in 2-3 sentences.'}])
        return r.choices[0].message.content.strip()
    except: return f"Found {len(idxs)} spectral matches."

def run_chat(text):
    c=client()
    if not c: return "Add Ollama credentials in the sidebar."
    if not st.session_state.tile_b64: return "Load a NAIP tile first."
    msgs=[{"role":"system","content":ANALYSIS_PROMPT}]
    for i,m in enumerate(st.session_state.messages):
        if m["role"]=="assistant":
            msgs.append({"role":"assistant","content":m["content"]})
        elif i==0:
            msgs.append({"role":"user","content":[
                {"type":"text","text":m["content"]},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{st.session_state.tile_b64}"}},
            ]})
        else: msgs.append({"role":"user","content":m["content"]})
    full=""; box=st.empty()
    try:
        for chunk in c.chat.completions.create(model=ollama_model,messages=msgs,stream=True):
            delta=chunk.choices[0].delta.content or ""
            full+=delta; box.markdown(full+"▌")
        box.markdown(full)
    except Exception as e: full=f"Error: {e}"; box.markdown(full)
    return full

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

# ── AOI selection ─────────────────────────────────────────────────────────────
map_tab, ll_tab = st.tabs(["🗺️ Draw on Map", "📍 Lat / Lon"])

with map_tab:
    st.caption("Draw a rectangle, then click Load.")
    dm=folium.Map(location=st.session_state.map_center,zoom_start=st.session_state.map_zoom,tiles=None)
    folium.TileLayer(ESRI_URL,attr=ESRI_ATTR,max_zoom=19).add_to(dm)
    folium.plugins.Draw(draw_options={"rectangle":True,"polyline":False,"polygon":False,
                                      "circle":False,"marker":False,"circlemarker":False},
                        position="topleft").add_to(dm)
    if st.session_state.aoi_bbox:
        w,s,e,n=st.session_state.aoi_bbox
        folium.Rectangle([[s,w],[n,e]],color="#58a6ff",weight=2,
                          fill=True,fill_color="#58a6ff",fill_opacity=0.1).add_to(dm)
    md=st_folium(dm,width="100%",height=320,returned_objects=["all_drawings"],key="draw_map")
    def parse_bbox(md):
        try:
            g=md.get("all_drawings") or []
            if not g: return None
            coords=g[-1]["geometry"]["coordinates"][0]
            lngs=[c[0] for c in coords]; lats=[c[1] for c in coords]
            return [min(lngs),min(lats),max(lngs),max(lats)]
        except: return None
    drawn=parse_bbox(md)
    if drawn and drawn!=st.session_state.aoi_bbox:
        if (drawn[2]-drawn[0])>0 and (drawn[3]-drawn[1])>0:
            st.session_state.aoi_bbox=drawn
    if st.session_state.aoi_bbox:
        w,s,e,n=st.session_state.aoi_bbox
        st.caption(f"AOI: W{w:.4f} S{s:.4f} E{e:.4f} N{n:.4f}")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🛰️ Load from AOI",disabled=(not st.session_state.aoi_bbox),
                     use_container_width=True,key="b_map"):
            with st.spinner("Fetching NAIP..."):
                try:
                    tb,b64,item=fetch_tile(st.session_state.aoi_bbox)
                    reset_tile(tb,b64,item)
                    w,s,e,n=st.session_state.aoi_bbox
                    st.session_state.map_center=[(s+n)/2,(w+e)/2]
                    st.success(f"Loaded {item.datetime.date()}")
                except Exception as e: st.error(str(e))

with ll_tab:
    c1,c2,c3=st.columns([2,2,1])
    with c1: lat=st.number_input("Lat",value=DEF_LAT,format="%.4f")
    with c2: lon=st.number_input("Lon",value=DEF_LON,format="%.4f")
    with c3: buf=st.slider("Buf°",0.001,0.01,0.003,step=0.001)
    if st.button("🛰️ Load from Point",use_container_width=True,key="b_ll"):
        with st.spinner("Fetching NAIP..."):
            try:
                tb,b64,item=fetch_tile([lon-buf,lat-buf,lon+buf,lat+buf])
                reset_tile(tb,b64,item)
                st.session_state.aoi_bbox=[lon-buf,lat-buf,lon+buf,lat+buf]
                st.session_state.map_center=[lat,lon]
                st.success(f"Loaded {item.datetime.date()}")
            except Exception as e: st.error(str(e))

st.markdown("---")

# ── Results section (full width, shown when search has run) ───────────────────
if st.session_state.res_chip_bytes:
    n = len(st.session_state.res_chip_bytes)
    st.markdown(f"### 🔍 Similar Tiles  ·  {n} results")
    st.caption("Query tile shown first (cyan), followed by the most similar locations found in the scene.")

    # One row: query + all results
    cols = st.columns(n + 1)
    cols[0].image(st.session_state.res_query_bytes, use_container_width=True)
    cols[0].markdown('<div class="tile-label query-label">◈ QUERY</div>', unsafe_allow_html=True)
    for k in range(n):
        cols[k+1].image(st.session_state.res_chip_bytes[k], use_container_width=True)
        cols[k+1].markdown(
            f'<div class="tile-label result-label">#{k+1} · {st.session_state.res_scores[k]:.3f}</div>',
            unsafe_allow_html=True)

    # Map
    if st.session_state.res_map_html:
        st.markdown("#### 📌 Locations on Map")
        components.html(st.session_state.res_map_html, height=320)

    # Spectral cards
    if st.session_state.res_spec_reports:
        st.markdown("#### 🌿 Spectral Summary")
        sr=st.session_state.res_spec_reports
        scols=st.columns(min(4,len(sr)))
        for k,r in enumerate(sr[:4]):
            def bar(name,val):
                pct=max(0,min(100,int((val+1)/2*100)))
                color=("#3fb950" if val>.3 else "#58a6ff" if val>0 else
                       "#d29922" if val>-.2 else "#f85149")
                return (f'<div style="display:flex;justify-content:space-between;font-size:.75rem;">'
                        f'<span style="color:#8b949e;">{name}</span>'
                        f'<span style="color:#e6edf3;font-weight:600;">{val:+.3f}</span></div>'
                        f'<div style="height:5px;background:#21262d;border-radius:3px;margin:2px 0 5px;">'
                        f'<div style="height:5px;width:{pct}%;background:{color};border-radius:3px;"></div></div>')
            with scols[k]:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;">'
                    f'<div style="font-size:.7rem;color:#8b949e;margin-bottom:6px;">#{k+1} · {r["nc"]}</div>'
                    f'{bar("NDVI",r["ndvi"])}{bar("NDWI",r["ndwi"])}'
                    f'{bar("EVI",r["evi"])}{bar("Brightness",r["brightness"]*2-1)}'
                    f'</div>', unsafe_allow_html=True)

    if st.session_state.res_label:
        st.info(st.session_state.res_label)
    st.markdown("---")

# ── Main imagery + chat ───────────────────────────────────────────────────────
img_col, chat_col = st.columns([1, 1], gap="large")

with img_col:
    st.markdown("#### NAIP Imagery")
    if st.session_state.tile_bytes:
        st.image(st.session_state.tile_bytes, use_container_width=True)
    else:
        st.info("Load a tile using the controls above.")

with chat_col:
    st.markdown("#### Ask anything")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"]=="user" and msg.get("intent"):
                i=msg["intent"]
                cls=("pill-spectral" if i=="SPECTRAL" else "pill-search" if i=="SEARCH" else "pill-chat")
                lbl=("🌿 SPECTRAL" if i=="SPECTRAL" else "🔍 SEARCH" if i=="SEARCH" else "💬 CHAT")
                st.markdown(f'<span class="pill {cls}">{lbl}</span>',unsafe_allow_html=True)
            st.markdown(msg["content"])

    prompt=st.chat_input("Ask a question, or describe what to find...",
                         disabled=(not st.session_state.tile_bytes))
    if prompt:
        with st.spinner("Routing..."): intent=classify_intent(prompt)
        st.session_state.messages.append({"role":"user","content":prompt,"intent":intent})

        with st.chat_message("user"):
            cls=("pill-spectral" if intent=="SPECTRAL" else "pill-search" if intent=="SEARCH" else "pill-chat")
            lbl=("🌿 SPECTRAL" if intent=="SPECTRAL" else "🔍 SEARCH" if intent=="SEARCH" else "💬 CHAT")
            st.markdown(f'<span class="pill {cls}">{lbl}</span>',unsafe_allow_html=True)
            st.markdown(prompt)

        item=st.session_state.tile_scene
        with st.chat_message("assistant"):
            if intent=="SPECTRAL" and item:
                response=run_spectral_search(prompt,item)
                st.markdown(response)
            elif intent=="SEARCH" and item:
                response=run_visual_search(prompt,item)
                st.markdown(response)
            else:
                response=run_chat(prompt)

        st.session_state.messages.append({"role":"assistant","content":response})
        st.session_state.res_label=response
        st.rerun()
