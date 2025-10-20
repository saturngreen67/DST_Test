import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import io
import re
import os
from functools import partial
import leafmap.foliumap as leafmap
import rasterio
from rasterio.io import MemoryFile
import rasterio.transform
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contextily as cx

st.set_page_config(page_title="My Data Dashboard", layout="centered")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/NATURE-DEMO/Decision_Support_Tool/main"
GITHUB_API_BASE = "https://api.github.com/repos/NATURE-DEMO/Decision_Support_Tool/contents/texts"

GITHUB_IMAGE_BASE_URL = f"{GITHUB_RAW_BASE}/images"
GITHUB_TIFF_URL = f"{GITHUB_RAW_BASE}/Koppen/1991-2020/koppen_geiger_0p1.tif"

items = [
    {"name": "Demo site 1-A", "address": "Lattenbach Valley, Austria", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo1.jpg", "github_key": "demo1a", "coordinate": [47.148472, 10.499805]},
    {"name": "Demo site 1-B", "address": "Brunntal, Austria", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo2.jpg", "github_key": "demo1b", "coordinate": [47.625027, 15.052111]},
    {"name": "Demo site 2", "address": "Brasov City, Romania", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo3.jpg", "github_key": "demo2", "coordinate": [45.647078, 25.593030]},
    {"name": "Demo site 3", "address": "Slovenia", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo4.jpg", "github_key": "demo3", "coordinate": [46.0345, 14.461]},
    {"name": "Demo site 4", "address": "Zvolen, Slovakia", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo5.jpg", "github_key": "demo4", "coordinate": [48.5707, 19.1462]},
    {"name": "Demo site 5", "address": "Globocica, Macedonia", "icon_url": f"{GITHUB_IMAGE_BASE_URL}/logo6.png", "github_key": "demo5", "coordinate": [48.5647, 19.114430]},
]
BACKGROUND_IMAGE_URL = f"{GITHUB_IMAGE_BASE_URL}/main_logo.png"

@st.cache_data(ttl=3600)
def cached_get(url: str) -> bytes:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content

@st.cache_data(ttl=3600)
def cached_json(url: str):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=3600)
def cached_text(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text

@st.cache_data(ttl=3600)
def cached_base64_image(url: str) -> str | None:
    try:
        b = cached_get(url)
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return None

@st.cache_data(ttl=3600)
def cached_read_excel(url: str) -> pd.DataFrame | None:
    try:
        b = cached_get(url)
        return pd.read_excel(io.BytesIO(b))
    except Exception:
        return None

@st.cache_data(ttl=600)
def list_github_folder(github_key: str):
    api_url = f"{GITHUB_API_BASE}/{github_key}/level1?ref=main"
    return cached_json(api_url)

@st.cache_data(ttl=600)
def get_sorted_txt_files(github_key: str):
    api_url = f"{GITHUB_API_BASE}/{github_key}?ref=main" 
    items_json = cached_json(api_url)
    if not isinstance(items_json, list):
        return []
    txts = [i for i in items_json if i.get("name", "").endswith(".txt")]
    def keyfn(f):
        m = re.match(r"^(\d+)", f["name"])
        return int(m.group(1)) if m else float("inf")
    txts.sort(key=keyfn)
    return txts

@st.cache_data(ttl=600)
def download_file_bytes(download_url: str) -> bytes:
    return cached_get(download_url)

def format_label(label: str) -> str:
    if isinstance(label, str) and label.startswith("CI"):
        return f"CI$_{{{label[2:]}}}$"
    return str(label)

def get_series_display_names():
    return {
        "CI": "Condition of the infrastructure (CI)",
        "CIH": "Condition of CI after exposure to hazard (CIH)",
        "CIHG": "Condition of CI after exposure to hazard but protected by GPI (CIHG)",
        "CIHN": "Condition of CI after exposure to hazard but protected by NbS (CIHN)",
        "CIHGN": "Condition of CI after exposure to hazard but protected by both GPI and NBS (CIHGN)"
    }

KOPPEN_COLORS = np.array([
    [0,0,255], [0,120,255], [70,170,250], [255,0,0], [255,150,150],
    [245,165,0], [255,220,100], [255,255,0], [200,200,0], [150,150,0],
    [150,255,150], [100,200,100], [50,150,50], [200,255,80], [100,255,80],
    [50,200,0], [255,0,255], [200,0,200], [150,50,150], [150,100,150],
    [170,175,255], [90,120,220], [75,80,180], [50,0,135], [0,255,255],
    [55,200,255], [0,125,125], [0,70,95], [178,178,178], [102,102,102]
]) / 255.0

KOPPEN_CLASSES = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc",
    20: "Dsd", 21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF"
}

def quick_koppen_map(tif_path, lat, lon):
    
    zoom = 1.0
    koppen_alpha = 0.6
    
    cmap = ListedColormap(KOPPEN_COLORS)
    class_labels = [KOPPEN_CLASSES[i] for i in range(1, 31)]
    
    min_lon, max_lon = lon - zoom, lon + zoom
    min_lat, max_lat = lat - zoom, lat + zoom
    
    try:
        response = requests.get(tif_path)
        if response.status_code == 200:
            tif_file = io.BytesIO(response.content)
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                row_min, col_min = src.index(min_lon, max_lat)
                row_max, col_max = src.index(max_lon, min_lat)
                row_start, row_end = sorted([row_min, row_max])
                col_start, col_end = sorted([col_min, col_max])
                data_cropped = data[row_start:row_end, col_start:col_end]
        else:
            st.error(f"Failed to download TIFF file from {tif_path}")
            return None
            
    except Exception as e:
        st.error(f"Map error: {e}")
        return None
    
    data_cropped = np.where(data_cropped == 0, np.nan, data_cropped)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_cropped, cmap=cmap, extent=(min_lon, max_lon, min_lat, max_lat), 
                    origin='upper', alpha=koppen_alpha, zorder=2)
    
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, alpha=0.8, zorder=1)
    
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title(f"Köppen-Geiger Climate Map")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = plt.colorbar(im, ticks=range(1, 31), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(class_labels, fontsize=8)
    
    return fig

def create_radar_chart_plotly(kpis_df: pd.DataFrame, selected_series: list, title: str):
    df = kpis_df.copy()
    categories = df.iloc[:, 0].astype(str).tolist()
    fig = go.Figure()
    for col in selected_series:
        vals = pd.to_numeric(df[col], errors="coerce").tolist()
        if not vals:
            continue
        vals_loop = vals + [vals[0]]
        cats_loop = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r=vals_loop, theta=cats_loop, fill="toself", name=col))
    fig.update_layout(polar=dict(radialaxis=dict(range=[1,5], tickvals=[1,2,3,4,5])), title=title, height=650)
    return fig

def create_kpi_analysis_plots_plotly(kpis_df: pd.DataFrame, el_df: pd.DataFrame, selected_item_name: str):
    df1 = kpis_df.set_index(kpis_df.columns[0])
    df2 = el_df.set_index(el_df.columns[0])
    figs = []
    max_pairs = min(4, max(0, len(df1.columns)-1), len(df2.columns))
    
    x_grid = np.linspace(0, 6, 60)
    y_grid = np.linspace(0, 6, 60)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_heatmap = (X + Y) / 2
    
    custom_colorscale = [
        [0.0, 'green'],    
        [0.5, 'yellow'],   
        [1.0, 'red']       
    ]

    def format_html_label(label: str) -> str:
        if isinstance(label, str) and label.startswith("CI") and len(label) > 2:
            subscript_part = label[2:]
            return f"CI<sub>{subscript_part}</sub>"
        return str(label)

    for i in range(max_pairs):
        col_f1 = df1.columns[i+1] if (i+1) < len(df1.columns) else df1.columns[-1]
        col_f2 = df2.columns[i] if i < len(df2.columns) else df2.columns[-1]
        x = pd.to_numeric(df2[col_f2], errors='coerce')
        y = pd.to_numeric(df1[col_f1], errors='coerce')
        
        df_plot = pd.DataFrame({
            "Extent of Loss": x, 
            "Condition": y, 
            "Original_Label": df1.index.map(str)
        }).dropna()

        df_plot["Display_Label"] = df_plot["Original_Label"].apply(format_html_label)

        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=Z_heatmap,
            x=x_grid,
            y=y_grid,
            colorscale=custom_colorscale,
            zmin=1, 
            zmax=5, 
            showscale=False, 
            hoverinfo='none' 
        ))

        if df_plot.empty:
            formatted_title_f1 = format_html_label(col_f1)
            fig.update_layout(title=f"{formatted_title_f1} vs {col_f2} (No valid data)", height=450)
            figs.append(fig)
            continue
        
        jitter_x = np.random.uniform(-0.15, 0.15, size=(len(df_plot),))
        jitter_y = np.random.uniform(-0.15, 0.15, size=(len(df_plot),))
        df_plot["Extent_j"] = df_plot["Extent of Loss"] + jitter_x
        df_plot["Condition_j"] = df_plot["Condition"] + jitter_y

        positions = ["top right", "bottom right"]
        df_plot["Text_Position"] = [positions[i % len(positions)] for i in range(len(df_plot))]

        scatter_trace = go.Scatter(
            x=df_plot["Extent_j"],
            y=df_plot["Condition_j"],
            mode='markers+text', 
            text=df_plot["Display_Label"], 
            textposition=df_plot["Text_Position"].tolist(), 
            textfont=dict(size=10), 
            marker=dict(size=12, line=dict(width=1, color="black")),
            name=f"{format_html_label(col_f1)} Data",
            hovertemplate="<b>Label:</b> %{customdata[2]}<br><b>Extent of Loss:</b> %{customdata[0]:.2f}<br><b>Condition:</b> %{customdata[1]:.2f}<extra></extra>",
            customdata=df_plot[["Extent of Loss", "Condition", "Display_Label"]].values
        )
        fig.add_trace(scatter_trace)

        formatted_title_f1 = format_html_label(col_f1) 
        fig.update_layout(
            title=f"<b>{formatted_title_f1}</b> vs Extent of Loss",
            xaxis=dict(
                dtick=1, 
                range=[0, 6], 
                title="Extent of Loss"
            ),
            yaxis=dict(
                dtick=1, 
                range=[0, 6], 
                title=formatted_title_f1 
            ),
            height=450,
            showlegend=False
        )
        
        figs.append(fig)
        
    return figs

@st.cache_data(ttl=600)
def get_kpis_excel(github_key: str) -> pd.DataFrame | None:
    url = f"{GITHUB_RAW_BASE}/texts/{github_key}/level1/1KPIs.xlsx"
    return cached_read_excel(url)

@st.cache_data(ttl=600)
def get_el_excel(github_key: str) -> pd.DataFrame | None:
    url = f"{GITHUB_RAW_BASE}/texts/{github_key}/level1/2el.xlsx"
    return cached_read_excel(url)

@st.cache_data(ttl=600)
def get_interpretation_text(github_key: str) -> str | None:
    url = f"{GITHUB_RAW_BASE}/texts/{github_key}/level1/interpretation.txt"
    try:
        return cached_text(url)
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_climate_report_text(github_key: str) -> str | None:
    url = f"{GITHUB_RAW_BASE}/texts/{github_key}/climate/climate_report.txt"
    try:
        return cached_text(url)
    except Exception:
        return None

bg_b64 = cached_base64_image(BACKGROUND_IMAGE_URL)
if bg_b64:
    st.sidebar.markdown(f'<img src="data:image/png;base64,{bg_b64}" style="width:100%;">', unsafe_allow_html=True)

st.sidebar.markdown("""
    <style>
        .custom-link { text-decoration: none; color: inherit;}
        .custom-button-container {
            width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2); transition: transform 0.12s;
            background-size: cover; background-position: center; color: white; text-shadow: 1px 1px 2px black;
        }
        .custom-button-container:hover { transform: scale(1.02); box-shadow: 4px 4px 10px rgba(0,0,0,0.25);}
    </style>
""", unsafe_allow_html=True)

for it in items:
    b64 = cached_base64_image(it["icon_url"])
    if b64:
        link_url = f"?item={it['github_key']}"
        html = f'''
        <a href="{link_url}" target="_self" class="custom-link">
            <div class="custom-button-container" style="background-image: url('data:image/png;base64,{b64}');">
                <h4 style="margin:0; padding:0; color:white;"><b>{it["name"]}</b></h4>
                <p style="margin:0; padding:0; font-size:14px; color:white;">{it["address"]}</p>
            </div>
        </a>
        '''
        st.sidebar.markdown(html, unsafe_allow_html=True)

query_params = st.query_params
selected_key = query_params.get("item", items[0]["github_key"])
items_map = {it["github_key"]: it for it in items}
selected_item = items_map.get(selected_key, items[0])

DEFAULT_CENTER = [41.500, 20.5308]
map_center = selected_item.get("coordinate", DEFAULT_CENTER)
map_zoom = 15

st.title(f"Risk assessment for {selected_item['name']}: {selected_item['address']}")

with st.container():
    with st.expander("Site Information and Maps"):
        with st.expander("Site Information"):
            st.markdown("""
                <style>
                .justified-text { text-align: justify; display:flex; flex-direction:column; justify-content:flex-end; min-height:100px; }
                </style>
            """, unsafe_allow_html=True)

            txt_files = get_sorted_txt_files(selected_item["github_key"])
            if not txt_files:
                st.warning("No text files found in the directory.")
            else:
                for f in txt_files:
                    name = f["name"]
                    display_name = os.path.splitext(name)[0]
                    title = display_name[1:] if len(display_name) > 1 and display_name[0].isdigit() else display_name
                    st.markdown(f'<div class="justified-text"><h1 style="font-size:30px;">{title}</h1></div>', unsafe_allow_html=True)
                    try:
                        b = download_file_bytes(f["download_url"])
                        txt = b.decode("utf-8")
                        st.markdown(f'<div class="justified-text">{txt.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                    except Exception:
                        st.error(f"Unable to load {name}")

        with st.expander("Maps"):
            m = leafmap.Map(center=map_center, zoom=map_zoom, height="700px")
            m.add_basemap("SATELLITE")
            m.add_marker(map_center, tooltip=selected_item["name"])
            m.to_streamlit()

            st.subheader("Köppen-Geiger Climate Classification")
            fig = quick_koppen_map(GITHUB_TIFF_URL, map_center[0], map_center[1])
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("Unable to display Köppen-Geiger Climate Map.")

            st.subheader("Climate Report")
            report = get_climate_report_text(selected_item["github_key"])
            if report:
                st.markdown(report)
            else:
                st.warning("Climate report not found for this site.")

    with st.container():
        with st.expander("Level 1"):
            with st.expander("Information tables"):
                try:
                    files_json = list_github_folder(selected_item["github_key"])
                    xlsx_files = [f for f in files_json if isinstance(f, dict) and f.get("name", "").endswith(".xlsx")]
                    def order_fn(name):
                        m = re.match(r'^(\d+)_', name)
                        return int(m.group(1)) if m else float("inf")
                    xlsx_files.sort(key=lambda f: order_fn(f["name"]))
                    for f in xlsx_files:
                        name = f["name"]
                        modified = re.sub(r'^\d+_', '', name[1:]).replace('.xlsx', '')
                        st.subheader(modified)
                        try:
                            b = download_file_bytes(f["download_url"])
                            df = pd.read_excel(io.BytesIO(b))
                            st.dataframe(df, hide_index=True)
                        except Exception as e:
                            st.error(f"Failed to download or read {name}: {e}")
                except Exception as e:
                    st.error(f"Failed to fetch directory listing: {e}")

            with st.expander("Perceived risk"):
                st.subheader("KPI Radar Chart")
                kpis_df = get_kpis_excel(selected_item["github_key"])
                if kpis_df is None or kpis_df.empty:
                    st.warning("No KPIs data found in the level1 folder.")
                else:
                    st.dataframe(kpis_df, use_container_width=True)
                    available_series = kpis_df.columns[1:].tolist()
                    display_names = get_series_display_names()
                    st.subheader("Select the CI conditions that you want to plot:")
                    selected_series = []
                    for s in available_series:
                        label = display_names.get(s, s)
                        if st.checkbox(label, value=True, key=f"checkbox_{s}"):
                            selected_series.append(s)
                    if selected_series:
                        radar_fig = create_radar_chart_plotly(kpis_df, selected_series, title=f"Radar Chart of {selected_item['name']} infrastructure")
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.warning("Please select at least one series to display.")

                st.subheader("KPI Analysis: Extent of Loss vs. CI Condition")
                el_df = get_el_excel(selected_item["github_key"])
                if (kpis_df is None or kpis_df.empty) or (el_df is None or el_df.empty):
                    st.warning("Both KPIs.xlsx and el.xlsx files are required for KPI analysis plots.")
                else:
                    st.dataframe(el_df, use_container_width=True)
                    figs = create_kpi_analysis_plots_plotly(kpis_df, el_df, selected_item["name"])
                    cols = st.columns(2)
                    for i, fig in enumerate(figs):
                        with cols[i % 2]:
                            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Interpretation"):
                interp = get_interpretation_text(selected_item["github_key"])
                if interp:
                    st.markdown(interp)
                else:
                    st.warning("No interpretation text found or error loading interpretation.")

        with st.expander("Level 2"):
            st.markdown("Under Construction")

        with st.expander("Level 3"):
            st.markdown("Under Construction")