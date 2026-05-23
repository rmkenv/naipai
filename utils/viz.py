"""
utils/viz.py — Folium map builders and Plotly UMAP scatter
"""
import numpy as np
import folium
import folium.plugins
import plotly.graph_objects as go
from shapely.geometry import mapping

import naip_config as cfg

ESRI_TILES = cfg.ESRI_TILES
ESRI_ATTR  = cfg.ESRI_ATTR


def _satellite_layer(name="Esri Satellite"):
    return folium.TileLayer(
        tiles=ESRI_TILES, attr=ESRI_ATTR, name=name, max_zoom=19, control=True)


def score_to_color(score, min_s, max_s):
    norm = (score - min_s) / (max_s - min_s + 1e-8)
    r = int(255 * (1 - norm))
    return f"#{r:02x}d23c"


def build_draw_map(center=(40.71, -74.0), zoom=12, existing_bbox=None):
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None)
    _satellite_layer().add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", control=True).add_to(m)
    folium.LayerControl().add_to(m)
    folium.plugins.Draw(
        draw_options={"rectangle": True, "polyline": False, "polygon": False,
                      "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": False},
        position="topleft",
    ).add_to(m)
    if existing_bbox:
        west, south, east, north = existing_bbox
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#58a6ff", weight=2,
            fill=True, fill_color="#58a6ff", fill_opacity=0.12,
            tooltip="Current AOI",
        ).add_to(m)
    return m


def build_result_map(query_idx, result_indices, result_scores, chip_gdf, center, zoom=14):
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None)
    _satellite_layer().add_to(m)
    folium.LayerControl().add_to(m)

    rows = chip_gdf.loc[chip_gdf["chip_id"] == query_idx, "geometry"]
    if not rows.empty:
        folium.GeoJson(
            mapping(rows.iloc[0]),
            style_function=lambda _: {"fillColor": "#00e5ff", "color": "#00e5ff",
                                       "weight": 2.5, "fillOpacity": 0.45},
            tooltip=folium.Tooltip(f"<b>QUERY</b> — chip #{query_idx}"),
        ).add_to(m)

    min_s = min(result_scores) if result_scores else 0
    max_s = max(result_scores) if result_scores else 1
    for rank, (idx, score) in enumerate(zip(result_indices, result_scores), 1):
        rows = chip_gdf.loc[chip_gdf["chip_id"] == idx, "geometry"]
        if rows.empty:
            continue
        color = score_to_color(score, min_s, max_s)
        folium.GeoJson(
            mapping(rows.iloc[0]),
            style_function=lambda _, c=color: {"fillColor": c, "color": "#ffffff",
                                                "weight": 1.5, "fillOpacity": 0.55},
            tooltip=folium.Tooltip(f"<b>Rank {rank}</b><br>Chip #{idx}<br>Sim: {score:.4f}"),
        ).add_to(m)
    return m


def build_umap_scatter(proj, query_idx, result_indices, result_scores, n_chips):
    result_set = set(result_indices)
    bg_mask = [i for i in range(n_chips) if i not in result_set and i != query_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proj[bg_mask, 0], y=proj[bg_mask, 1], mode="markers",
        marker=dict(size=4, color="#3a3f47", opacity=0.5), name="All chips",
        hovertemplate="chip %{text}<extra></extra>",
        text=[str(i) for i in bg_mask],
    ))
    min_s = min(result_scores) if result_scores else 0
    max_s = max(result_scores) if result_scores else 1
    norm_scores = [(s - min_s) / (max_s - min_s + 1e-8) for s in result_scores]
    fig.add_trace(go.Scatter(
        x=proj[result_indices, 0], y=proj[result_indices, 1], mode="markers",
        marker=dict(size=10, color=norm_scores, colorscale="YlGn",
                    colorbar=dict(title="Similarity", thickness=12),
                    line=dict(width=1, color="white")),
        name="Similar chips",
        hovertemplate="chip %{text}<br>sim: %{customdata:.4f}<extra></extra>",
        text=[str(i) for i in result_indices], customdata=result_scores,
    ))
    fig.add_trace(go.Scatter(
        x=[proj[query_idx, 0]], y=[proj[query_idx, 1]], mode="markers",
        marker=dict(size=16, symbol="star", color="#00e5ff",
                    line=dict(width=1.5, color="white")),
        name="Query",
        hovertemplate=f"QUERY — chip {query_idx}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Space Mono, monospace"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text="Embedding Space (UMAP)", font=dict(size=13)),
        height=420,
    )
    return fig
