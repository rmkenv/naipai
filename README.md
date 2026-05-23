# 🛰️ NAIP Intelligence Platform

A unified Streamlit app combining two NAIP tools into one tabbed interface.

| Tab | What it does |
|-----|-------------|
| **💬 VLM Chat** | Fetch a NAIP tile by lat/lon and chat with a vision-language model (Qwen3-VL via Ollama Cloud) |
| **🔍 Similarity Search** | Draw an AOI, chip the scene, embed with ResNet-50, and find visually similar locations via FAISS cosine search |

**Data**: USDA NAIP via Microsoft Planetary Computer STAC  
**Embed model**: ResNet-50 (ImageNet pretrained, 2048-d)  
**Index**: FAISS `IndexFlatIP` (cosine similarity on L2-normalized vectors)  
**VLM**: Any Ollama-compatible vision model (default: `qwen3-vl:7b`)

---

## Setup

```bash
pip install -r requirements.txt
```

### Secrets (Ollama Cloud — needed for Chat tab only)

Create `.streamlit/secrets.toml`:

```toml
OLLAMA_HOST      = "https://ollama.com"
OLLAMA_API_KEY   = "your-key-here"
OLLAMA_MODEL     = "qwen3-vl:7b"
```

Or export as environment variables:

```bash
export OLLAMA_HOST="https://ollama.com"
export OLLAMA_API_KEY="your-key-here"
export OLLAMA_MODEL="qwen3-vl:7b"
```

### Run

```bash
streamlit run app.py
```

---

## Project structure

```
naip_combined/
├── app.py              # Main tabbed Streamlit app
├── config.py           # All tuneable constants
├── requirements.txt
└── utils/
    ├── embeddings.py   # ResNet-50 pipeline + FAISS helpers
    ├── imagery.py      # NAIP STAC fetch, chipping, geo utils
    └── viz.py          # Folium map builders + Plotly UMAP scatter
```

---

## Origins

Merged from:
- [`naipchat`](https://github.com/rmkenv/naipchat) — NAIP × VLM chat
- [`openembed`](https://github.com/rmkenv/openembed) — NAIP embeddings similarity search
