import streamlit as st
import requests
import folium
import folium.plugins
from streamlit_folium import st_folium
from shapely.geometry import Polygon
import pandas as pd
import json
import re
import time
import os
from google import genai
from google.genai.errors import APIError
import numpy as np
import io
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contextily as cx
import traceback
import plotly.graph_objects as go

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

KOPPEN_TIFF_URL = "https://raw.githubusercontent.com/saturngreen67/streamlit_tests/main/Koppen/1991-2020/koppen_geiger_0p1.tif"

def polygon_style_function(feature):
    return {'fillColor': 'blue', 'color': 'blue'}

def generate_koppen_map_plot(lat, lon, zoom_range=1.0):
    
    cmap = ListedColormap(KOPPEN_COLORS)
    class_labels = [KOPPEN_CLASSES[i] for i in range(1, 31)]
    
    min_lon, max_lon = lon - zoom_range, lon + zoom_range
    min_lat, max_lat = lat - zoom_range, lat + zoom_range
    
    try:
        response = requests.get(KOPPEN_TIFF_URL, timeout=30)
        response.raise_for_status()
        
        with rasterio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            row_min, col_min = src.index(min_lon, max_lat)
            row_max, col_max = src.index(max_lon, min_lat)
            
            row_start, row_end = sorted([row_min, row_max])
            col_start, col_end = sorted([col_min, col_max])
            
            data_cropped = data[row_start:row_end, col_start:col_end]

    except requests.exceptions.RequestException as e:
        return f"Error downloading Köppen TIFF: {e}", None
    except Exception as e:
        return f"Error processing Köppen map data: {e}", None
    
    data_cropped = np.where(data_cropped == 0, np.nan, data_cropped)
    
    koppen_code = "N/A"
    if data_cropped.size > 0:
        center_row = (data_cropped.shape[0]) // 2
        center_col = (data_cropped.shape[1]) // 2
        
        center_code = int(data_cropped[center_row, center_col]) if not np.isnan(data_cropped[center_row, center_col]) else None
        koppen_code = KOPPEN_CLASSES.get(center_code, "N/A") if center_code else "N/A"
            
    else:
        koppen_code = "N/A"

    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        data_cropped, 
        cmap=cmap, 
        extent=(min_lon, max_lon, min_lat, max_lat),
        origin='upper', 
        alpha=0.6, 
        zorder=2
    )
    
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, alpha=0.8, zorder=1)
    except Exception:
        pass 
        
        
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title(f"Köppen-Geiger Climate Classification (Center: {koppen_code})")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    cbar = plt.colorbar(im, ticks=range(1, 31), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(class_labels, fontsize=8)
    
    return fig, koppen_code

def generate_context_report(center_lat, center_lon, area_sq_km, elements):
    if not st.session_state.get("gemini_client"):
        return "Gemini client not initialized. Cannot generate report."

    center_coord_str = f"{center_lat:.4f}, {center_lon:.4f}"
    
    infra_counts = {}
    for element in elements:
        tags = element.get('tags', {})
        for tag_key, tag_value in tags.items():
            if tag_key in ['building', 'highway', 'railway', 'water', 'power', 'amenity', 'leisure', 'natural']:
                category = tag_key.capitalize()
                infra_counts[category] = infra_counts.get(category, 0) + 1
                break
    
    extracted_infrastructure_list = "\n".join(
        [f"- {k}: {v} items" for k, v in sorted(infra_counts.items(), key=lambda item: item[1], reverse=True)]
    )
    if not extracted_infrastructure_list:
        extracted_infrastructure_list = "- No specific infrastructure elements found using the simple filters."

    system_instruction = (
        "You are an expert geographical and infrastructure analyst. Your task is to generate a report "
        "by analyzing the provided OpenStreetMap data and geographical coordinate. "
        "**You must use the Google Search tool** to find contextual data, elevation, and topography for the given coordinate. "
        "Follow the specified structured output format strictly. DO NOT include any climate or weather information."
    )
    
    user_prompt = f"""
    Analyze the following geographical data for a selected area:

    1. **Geographical Area Details:**
        - **Approximate Center Coordinate (Latitude, Longitude):** {center_coord_str}
        - **Approximate Area:** {area_sq_km:.2f} square kilometers
        
    2. **Extracted OpenStreetMap Infrastructure Data (Summary of {len(elements)} Items):**
    {extracted_infrastructure_list}

    **REPORT INSTRUCTIONS:**
    Please use the coordinate and the extracted infrastructure details to search the internet for more contextual information about this geographical area.
    
    **Provide the report in the following structured format:**...

    """
    
    try:
        response = st.session_state["gemini_client"].models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config={
                "system_instruction": system_instruction, 
                "tools": [{"google_search": {}}] 
            }
        )
        
        return response.text

    except APIError as e:
        return f"Gemini API Error (Context Report): {e}."
    except Exception as e:
        st.error(f"An unexpected error occurred during context report generation: {e}")
        return "An unexpected error occurred during context report generation."


def generate_koppen_interpretation(koppen_code):
    if not st.session_state.get("gemini_client"):
        return "Gemini client not initialized. Cannot generate climate interpretation."
    
    if koppen_code in ["N/A", "Unknown"]:
        return "Cannot generate interpretation. The Köppen climate code could not be determined from the map."

    system_instruction = (
        "You are an expert climatologist. Your task is to provide a detailed, easy-to-understand interpretation "
        "of the given Köppen Climate Classification code. **You must use the Google Search tool** "
        "to find detailed climate conditions. Ensure all temperatures are provided in Celcius. "
        "The response must focus purely on climate conditions and meaning."
    )
    
    user_prompt = f"""
    Provide a detailed interpretation of the following Köppen Climate Classification code: **{koppen_code}**.

    Your report must cover:
    1.  **Full Classification Name:** (e.g., 'Humid subtropical climate').
    2.  **Key Characteristics:** The general weather patterns, typical seasonal changes, and defining temperature and precipitation features (e.g., hot/cold summers, dry/wet winters).
    3.  **Ecological Implications:** Briefly mention the types of vegetation or agriculture typically found in this climate.
    
    Structure the answer logically using headings and bullet points.
    """
    
    try:
        response = st.session_state["gemini_client"].models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config={
                "system_instruction": system_instruction, 
                "tools": [{"google_search": {}}] 
            }
        )
        
        return response.text

    except APIError as e:
        return f"Gemini API Error (Köppen Interpretation): {e}."
    except Exception as e:
        st.error(f"An unexpected error occurred during Köppen interpretation generation: {e}")
        return "An unexpected error occurred during Köppen interpretation generation."

def generate_risk_interpretation(df_risks: pd.DataFrame, kpis: list, scenarios: dict):
    if not st.session_state.get("gemini_client"):
        return "Gemini client not initialized. Cannot generate risk interpretation."

    df_risks_prompt = df_risks.rename_axis('').to_markdown()

    scenario_desc = "\n".join([f"- **{abbr}**: {desc}" for abbr, desc in scenarios.items()])
    kpi_list = "\n".join([f"- {k}" for k in kpis])

    system_instruction = (
        "You are an expert risk and resilience analyst. Your task is to interpret a stakeholder "
        "risk assessment matrix. The ratings are from 1 (best condition/lowest risk) to 5 (worst condition/highest risk). "
        "**You must use the Google Search tool** to find contextual information related to 'critical infrastructure resilience' and 'risk assessment' to enrich your analysis. "
        "Your interpretation should compare the different scenarios and highlight the biggest perceived risks and the effectiveness of mitigation measures."
    )

    user_prompt = f"""
    Analyze the following risk matrix provided by stakeholders. The ratings are for Key Performance Indicators (KPIs) across different scenarios, where **1 is the best condition (lowest risk) and 5 is the worst condition (highest risk)**.

    **KPIs (Rows):**
    {kpi_list}

    **Scenarios (Columns - representing different protection measures):**
    {scenario_desc}
    
    **Risk Assessment Matrix (Ratings 1-5):**
    {df_risks_prompt}

    **REPORT INSTRUCTIONS:**
    Provide a concise, professional interpretation covering:
    1.  **General Observation:** What is the average perception of risk across all scenarios?
    2.  **Worst Scenarios:** Which scenario(s) have the highest overall risk perception (highest average rating)? Why?
    3.  **Key Vulnerabilities:** Which KPI(s) consistently show the highest risk rating (5 or 4) across multiple scenarios?
    4.  **Mitigation Effectiveness:** Interpret the perceived effectiveness of protection measures (comparing CI_H, CI_HG, CI_HN, CI_HNG). For example, do 'Grey' or 'Nature-based' solutions appear to significantly reduce the risk compared to the 'Hazard only' scenario (CI_H)?
    
    Structure the answer logically using professional headings and bullet points.
    """

    try:
        response = st.session_state["gemini_client"].models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config={
                "system_instruction": system_instruction, 
                "tools": [{"google_search": {}}] 
            }
        )
        
        return response.text

    except APIError as e:
        return f"Gemini API Error (Risk Interpretation): An issue occurred connecting to the service: {e}."
    except Exception as e:
        return f"An unexpected error occurred during risk interpretation generation: {e}"

@st.cache_data(ttl=3600)
def build_folium_map_object(center, zoom, polygon_data, drawing_key):
    
    m = folium.Map(location=center, 
                    zoom_start=zoom, 
                    tiles="CartoDB positron")

    folium.raster_layers.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/SDXw0l5jQ3C1QO7x/arcgis/rest/services/Koeppen_Geiger_Climate_Classification_2020/MapServer/tile/{z}/{y}/{x}',
        attr='Köppen-Geiger / Esri',
        name='Köppen-Geiger Climate Classification (Overlay)',
        overlay=True,
        opacity=0.6,
        control=True,
    ).add_to(m)

    draw = folium.plugins.Draw(export=False, draw_options={'polygon': True, 'rectangle': True})
    draw.add_to(m)
    folium.LayerControl().add_to(m)

    if polygon_data:
        folium.GeoJson(polygon_data, name="Drawn Polygon", 
                        style_function=polygon_style_function).add_to(m)
    
    return m 

@st.cache_data(ttl=3600)
def geocode_location(location_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': location_name, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'GeneralDecisionSupportTool/1.0'}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            return [lat, lon]
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Geocoding error: {e}")
        return None

def reset_polygon():
    st.session_state["last_polygon"] = None
    st.session_state["drawing_key"] += 1 
    st.session_state["extract_clicked"] = False
    st.session_state["extracted_data"] = None
    st.rerun()

def get_polygon_coords(geo_json):
    coords = geo_json["geometry"]["coordinates"][0]
    return [(lat, lon) for lon, lat in coords]

infra_options = {
    "Buildings": ['["building"]'], "Roads": ['["highway"]'], "Railways": ['["railway"]'],
    "Water": ['["water"]', '["waterway"]'], "Power": ['["power"]'], "Landuse": ['["landuse"]'],
    "Man-made Structures": ['["man_made"]'], "Barriers": ['["barrier"]'], "Natural Features": ['["natural"]'],
    "Amenities": ['["amenity"]'], "Leisure": ['["leisure"]'], "Dams & Waterworks": ['["waterway"="dam"]'],
}

def build_query(coords, selected_infras):
    
    coord_str = " ".join([f"{lat} {lon}" for lat, lon in coords])
    filters = []
    for infra_name in selected_infras:
        tag_filters = infra_options[infra_name] 
        for tag_filter in tag_filters:
            filters.append(f'nwr{tag_filter.strip()}(poly:"{coord_str}");')
            
    query_body = "\n".join(filters)
    return f"[out:json][timeout:90];\n(\n{query_body}\n);\nout body geom;"

def make_overpass_request(query, max_retries=2):
    overpass_url = "https://overpass-api.de/api/interpreter"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=180)
            if response.status_code == 200: return response
            elif response.status_code == 400: st.error("⚠️ Overpass API Error: Bad Request (400). Check query syntax."); return response
            elif response.status_code == 429: st.error("❌ **Overpass API Error: Too Many Requests (429).**"); return response
            elif response.status_code == 504:
                if attempt < max_retries: time.sleep(5); continue
                else: st.error("⚠️ Overpass API Error: Server timeout (504)."); return response
            else: st.error(f"⚠️ Overpass API Error: HTTP Status Code {response.status_code}"); return response
        except requests.exceptions.Timeout:
            if attempt < max_retries: time.sleep(5); continue
            else: st.error("⚠️ Request timeout after multiple attempts"); return None
        except requests.exceptions.RequestException as e: st.error(f"⚠️ Network error: {str(e)}"); return None
    return None

def element_matches_infrastructure(element, infra_keys):
    if 'tags' not in element or not element['tags']: return False
    for key in infra_keys:
        if key in element['tags']: return True
    return False

def create_detailed_dataframe(elements):
    if not elements: return pd.DataFrame()
    data_rows = []
    for element in elements:
        row_data = {'type': element.get('type', ''), 'id': element.get('id', '')}
        if 'tags' in element and element['tags']:
            for tag_key, tag_value in element['tags'].items():
                row_data[f'tag.{tag_key}'] = tag_value
        data_rows.append(row_data)
    return pd.DataFrame(data_rows).fillna('')

def create_radar_chart_plotly(kpis_df: pd.DataFrame, selected_series: list, title: str):
    
    df = kpis_df.copy()
    if df.columns.size < 2:
        return None
    categories = df.iloc[:, 0].astype(str).tolist()
    fig = go.Figure()
    for col in selected_series:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").tolist()
        if len(vals) == 0:
            continue
        vals_loop = vals + [vals[0]]
        cats_loop = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r=vals_loop, theta=cats_loop, fill="toself", name=col))
    fig.update_layout(polar=dict(radialaxis=dict(range=[1,5], tickvals=[1,2,3,4,5])), title=title, height=650)
    return fig


def update_risk_matrix():
    edited_data = st.session_state["risk_matrix_editor"]
    if isinstance(edited_data, pd.DataFrame):
        st.session_state["risk_matrix_data"] = edited_data.to_dict()
        st.session_state["interpretation_report"] = ""

if "map_center" not in st.session_state:
    st.session_state["map_center"] = [51.1657, 10.4515]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 6
if "drawing_key" not in st.session_state:
    st.session_state["drawing_key"] = 0
if "last_polygon" not in st.session_state:
    st.session_state["last_polygon"] = None
if "extract_clicked" not in st.session_state:
    st.session_state["extract_clicked"] = False
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None

kpis = [
    "Safety, Reliability and Security (SRS)", 
    "Availability and Maintainability (AM)", 
    "Economy (EC)", 
    "Environment (EV)", 
    "Health and Politics (HP)"
]
scenarios = {
    "CI": "Current condition of the critical infrastructure",
    "CI_H": "Condition after natural hazard (H)",
    "CI_HG": "Condition after hazard but protected by grey measures (HG)",
    "CI_HN": "Condition after hazard but protected by nature-based solutions (HN)",
    "CI_HNG": "Condition after hazard but protected by both grey and nature-based solutions (HNG)"
}

initial_data = {scenario_key: {k: 1 for k in kpis} for scenario_key in scenarios}
if "risk_matrix_data" not in st.session_state:
    st.session_state["risk_matrix_data"] = pd.DataFrame(initial_data, index=kpis).to_dict()

if "last_radar_plot" not in st.session_state:
    st.session_state["last_radar_plot"] = None

if "interpretation_report" not in st.session_state:
    st.session_state["interpretation_report"] = ""

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
except (KeyError, AttributeError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        st.session_state["gemini_client"] = client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.session_state["gemini_client"] = None
else:
    st.warning("⚠️ GEMINI_API_KEY not found. AI report feature disabled. Please set the key.")
    st.session_state["gemini_client"] = None

st.set_page_config(page_title="General Decision Support Tool", layout="centered")
st.title("General Decision Support Tool")


with st.sidebar:
    st.image('https://raw.githubusercontent.com/NATURE-DEMO/Decision_Support_Tool/main/images/main_logo.png', width=300)

with st.expander("Information Extraction and Mapping"):
    st.markdown(
        """
        Use the search box to find a location, then draw a polygon on the map and click **"Extract Information"** below the map.
        """
    )

    st.header("Select Infrastructure Types")
    cols = st.columns(4)
    checkbox_states = {}
    all_infra_keys = list(infra_options.keys())

    for i, infra_name in enumerate(all_infra_keys):
        col_index = i % 4
        with cols[col_index]:
            if f"check_{infra_name}" not in st.session_state:
                default_state = infra_name in ["Buildings", "Roads", "Water"]
                st.session_state[f"check_{infra_name}"] = default_state
                
            checkbox_states[infra_name] = st.checkbox(
                infra_name, 
                value=st.session_state[f"check_{infra_name}"], 
                key=f"check_{infra_name}"
            )

    selected_infras = [k for k, is_selected in checkbox_states.items() if is_selected]
    if len(selected_infras) > 5:
        st.warning("⚠️ Selecting many infrastructure types may cause timeouts for large areas.")
    st.markdown("---") 

    st.header("Search Location and Draw Polygon")
    search_col, _ = st.columns([3, 1])

    with search_col:
        location_name = st.text_input("Search for a location to center the map:", value="Berlin, Germany")
        if st.button("Go to Location 🗺️"):
            with st.spinner(f"Geocoding '{location_name}'..."):
                coords = geocode_location(location_name)
                if coords:
                    st.session_state["map_center"] = coords
                    st.session_state["map_zoom"] = 12
                    st.session_state["drawing_key"] += 1 
                    st.success(f"Map centered on {location_name}. Now draw a polygon.")
                else:
                    st.error(f"Could not find coordinates for '{location_name}'.")

    map_object = build_folium_map_object(
        st.session_state["map_center"],
        st.session_state["map_zoom"],
        st.session_state["last_polygon"],
        st.session_state["drawing_key"]
    )

    output = st_folium(
        map_object, 
        height=600, 
        width=1200, 
        key=st.session_state["drawing_key"]
    )

    if output and output.get("last_active_drawing") and output["last_active_drawing"].get("geometry"):
        if output["last_active_drawing"].get("geometry").get("type") in ["Polygon", "Rectangle"]:
            if st.session_state["last_polygon"] != output["last_active_drawing"]:
                st.session_state["last_polygon"] = output["last_active_drawing"]
                st.session_state["extract_clicked"] = False 
        elif output["last_active_drawing"].get("geometry").get("coordinates") is None:
            st.session_state["last_polygon"] = None

    st.markdown("---")
    button_col, reset_col, _ = st.columns([2, 1, 3])

    with button_col:
        if st.button("Extract Information", type="primary", key="extract_btn", help="Trigger data extraction and AI report generation."):
            if st.session_state["last_polygon"] is None:
                st.error("Please draw a polygon on the map first.")
            else:
                st.session_state["extract_clicked"] = True
                st.session_state["extracted_data"] = None 
                st.rerun()

    with reset_col:
        if st.button("Reset Polygon", help="Clear the current drawn polygon.", key="reset_poly_btn"):
            reset_polygon()

    if st.session_state["extract_clicked"]:
        
        if st.session_state["extracted_data"] is None:
            
            geo_json = st.session_state["last_polygon"]
            
            try:
                coords = get_polygon_coords(geo_json)
                polygon = Polygon(coords)
                area_sq_km = polygon.area * (111**2) 
                
                center_lat = sum(c[0] for c in coords) / len(coords)
                center_lon = sum(c[1] for c in coords) / len(coords)
                
                query = build_query(coords, selected_infras)
                with st.spinner("Retrieving data from OpenStreetMap..."):
                    response = make_overpass_request(query)
                    if response is None or response.status_code != 200:
                        st.session_state["extract_clicked"] = False
                        st.stop()
                    data = response.json()
                elements = data.get("elements", [])
                
                if not elements:
                    st.warning("No data found in the selected area for the chosen types.")
                
                with st.spinner("Analyzing climate map data..."):
                    _, center_koppen_code = generate_koppen_map_plot(center_lat, center_lon)

                context_report = ""
                if st.session_state.get("gemini_client"):
                    with st.spinner(f"Generating Geographical & Infrastructure Report (Internet Search)..."):
                        context_report = generate_context_report(center_lat, center_lon, area_sq_km, elements)

                koppen_report = ""
                if st.session_state.get("gemini_client"):
                    with st.spinner(f"Generating Köppen Interpretation Report for code {center_koppen_code} (Internet Search)..."):
                        koppen_report = generate_koppen_interpretation(center_koppen_code)
                
                st.session_state["extracted_data"] = {
                    "elements": elements, "coords": coords, "area_sq_km": area_sq_km,
                    "context_report": context_report,
                    "koppen_report": koppen_report,
                    "koppen_code": center_koppen_code,
                    "center_lat": center_lat, "center_lon": center_lon
                }

            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred during extraction: {str(e)}")
                st.code(traceback.format_exc())
                st.session_state["extract_clicked"] = False
                st.stop()
        
        if st.session_state["extracted_data"]:
            elements = st.session_state["extracted_data"].get("elements", [])
            area_sq_km = st.session_state["extracted_data"].get("area_sq_km", 0)
            context_report = st.session_state["extracted_data"].get("context_report", "")
            koppen_report = st.session_state["extracted_data"].get("koppen_report", "")
            center_lat = st.session_state["extracted_data"].get("center_lat")
            center_lon = st.session_state["extracted_data"].get("center_lon")

            if st.session_state["extracted_data"].get("elements"):
                st.success(f"Successfully processed {len(elements)} OSM items ✅ (Area: {area_sq_km:.4f} km²)")
            
            st.markdown("---")
            st.subheader("🤖 Geographical & Infrastructure Context Report (Gemini)")
            if context_report:
                st.markdown(context_report)
            else:
                st.warning("The Geographical & Infrastructure Report failed to generate or the AI feature is disabled.")

            st.markdown("---")
            st.subheader("🗺️ Köppen-Geiger Climate Classification Map (Visual)")
            
            if center_lat is not None and center_lon is not None:
                plot_result, _ = generate_koppen_map_plot(center_lat, center_lon)
                
                if isinstance(plot_result, str):
                    st.error(plot_result) 
                else:
                    st.pyplot(plot_result)
            else:
                st.warning("Cannot display Köppen map: Center coordinates for the drawn polygon could not be determined.")

            st.markdown("---")
            st.subheader("☀️ Climate Interpretation Report (Gemini)")
            if koppen_report:
                st.markdown(koppen_report)
            else:
                st.warning("The Climate Interpretation Report failed to generate or the AI feature is disabled.")

            with st.expander("🔍 View Extracted Infrastructure Data Tables (OpenStreetMap Raw Data)"):
                st.subheader("📊 Detailed Infrastructure Data Tables")
                has_data_for_any_infra = False
                
                for infra in selected_infras:
                    keys_to_check = set()
                    for filter_str in infra_options[infra]:
                        key_match = re.search(r'\["(.+?)"', filter_str)
                        if key_match: keys_to_check.add(key_match.group(1))

                    infra_elements = [
                        element for element in elements 
                        if element_matches_infrastructure(element, keys_to_check)
                    ]
                    
                    if infra_elements:
                        has_data_for_any_infra = True
                        infra_df = create_detailed_dataframe(infra_elements)
                        st.subheader(f"{infra} ({len(infra_elements)} infrastructure items)")
                        st.dataframe(infra_df[[col for col in infra_df.columns if not col.startswith('geometry')]].head(15), width=1200) 
                        
                if not has_data_for_any_infra:
                    st.info("No detailed data to display for the selected infrastructure types.")


with st.expander("Level 1"):
    
    st.header("Perceived Risks Assessment 📊")

    kpis = [
        "Safety, Reliability and Security (SRS)", 
        "Availability and Maintainability (AM)", 
        "Economy (EC)", 
        "Environment (EV)", 
        "Health and Politics (HP)"
    ]
    scenarios = {
        "CI": "Current condition of the critical infrastructure",
        "CI_H": "Condition after natural hazard (H)",
        "CI_HG": "Condition after hazard but protected by grey measures (HG)",
        "CI_HN": "Condition after hazard but protected by nature-based solutions (HN)",
        "CI_HNG": "Condition after hazard but protected by both grey and nature-based solutions (HNG)"
    }

    df_data = st.session_state["risk_matrix_data"]
    df = pd.DataFrame(df_data, index=kpis)
    df.index.name = "KPI / Indicator"

    st.subheader("Stakeholders' Opinion About Infrastructure Condition")
    st.info("Please provide integers between **1 (best condition)** and **5 (worst condition)** for each cell. Values outside this range are not allowed.")

    matrix_tab, scenario_key_tab = st.tabs(["📊 Risk Matrix Input (1-5)", "📝 Scenario & KPI Definitions"])

    with scenario_key_tab:
        st.markdown("### Key Performance Indicators (KPIs)")
        st.markdown("""
        These indicators cover various dimensions of risk and resilience:
        * **SRS:** Safety, Reliability, and Security.
        * **AM:** Availability and Maintainability.
        * **EC:** Economy (cost, efficiency).
        * **EV:** Environment (environmental impact, sustainability).
        * **HP:** Health and Politics (public health, political stability).
        """)
        
        st.markdown("### Scenario Definitions")
        for abbr, desc in scenarios.items():
            html_abbr = abbr.replace("CI_HNG", "CI<sub>HNG</sub>").replace("CI_HN", "CI<sub>HN</sub>").replace("CI_HG", "CI<sub>HG</sub>").replace("CI_H", "CI<sub>H</sub>").replace("CI", "CI")
            st.markdown(f"**{html_abbr}** ({desc})", unsafe_allow_html=True)

    with matrix_tab:
        st.markdown("### Input Risks Rating (1 to 5)")
        
        column_config = {
            "KPI / Indicator": st.column_config.TextColumn(
                "KPI / Indicator",
                disabled=True
            )
        }
        
        for key, desc in scenarios.items():
            column_config[key] = st.column_config.NumberColumn(
                label=key,
                help=desc,
                min_value=1,
                max_value=5,
                default=1,
                format="%d",
                width="small"
            )

        st.data_editor(
            df,
            column_config=column_config,
            num_rows="fixed",
            key="risk_matrix_editor",
            on_change=update_risk_matrix
        )
        

        

        st.markdown("---")
        st.subheader("🔷 Radar Plot of Input Risks")
        try:
            kpis_for_plot = pd.DataFrame(st.session_state["risk_matrix_data"], index=kpis).reset_index()
        except Exception:
            kpis_for_plot = pd.DataFrame(df).reset_index()


        available_series = kpis_for_plot.columns[1:].tolist()

        if not available_series:
            st.info("No scenario columns available to plot. Please configure the risk matrix columns.")
        else:
            st.markdown("Select scenarios to include in the radar plot:")
            selected_series = []
            for s in available_series:
                checked = st.checkbox(s, value=True, key=f"radar_checkbox_{s}")
                if checked:
                    selected_series.append(s)
            
            if st.button("Plot", key="plot_radar_btn"):
                if not selected_series:
                    st.warning("Please select at least one scenario/column to plot.")
                else:
                    try:
                        radar_fig = create_radar_chart_plotly(kpis_for_plot, selected_series, title="Risk Radar - Input Ratings (1-5)")
                        if radar_fig is None:
                            st.error("Unable to generate radar figure. Check your input format.")
                        else:
                            st.session_state["last_radar_plot"] = radar_fig
                            st.plotly_chart(radar_fig, use_container_width=True, key="new_radar_plot")
                    except Exception as e:
                        st.error(f"Failed to create radar plot: {e}")
                        st.exception(e)
        
        if st.session_state["last_radar_plot"]:
             st.plotly_chart(st.session_state["last_radar_plot"], use_container_width=True, key="stored_radar_plot")

        st.markdown("---")
        with st.expander("Interpretation"):
            
            if st.session_state.get("gemini_client"):
                
                if st.button("Generate Interpretation Report 🤖", type="primary", help="Analyze the current risk matrix using Gemini with contextual search."):
                    
                    if not 'risk_matrix_data' in st.session_state:
                        st.error("Please populate the Risk Matrix table first before generating an interpretation.")
                        st.stop()
                        
                    try:
                        current_df = pd.DataFrame(st.session_state["risk_matrix_data"], index=kpis) 
                    except (KeyError, ValueError) as e:
                        st.error(f"Error reading risk matrix data: {e}. Ensure 'kpis' and 'risk_matrix_data' are correctly structured.")
                        current_df = None
                    
                    if current_df is not None:
                        with st.spinner("Generating Risk Matrix Interpretation (Gemini with Google Search)..."):
                            interpretation_report = generate_risk_interpretation(current_df, kpis, scenarios)
                        
                        st.session_state["interpretation_report"] = interpretation_report
                        
                        st.subheader("🤖 Risk Matrix Interpretation Report (Gemini)")
                        if interpretation_report:
                            st.markdown(interpretation_report)
                        else:
                            st.warning("The Risk Matrix Interpretation Report failed to generate.")
                
                if st.session_state["interpretation_report"]:
                    st.subheader("🤖 Risk Matrix Interpretation Report (Gemini)")
                    st.markdown(st.session_state["interpretation_report"])
                else:
                    st.info("Click the button above to generate the AI interpretation report based on the current matrix data.")

            else:
                st.warning("⚠️ Gemini client not initialized. AI interpretation feature disabled. Ensure GEMINI_API_KEY is available.")


with st.expander("Level 2"):
    st.write("Under construction")


with st.expander("Level 3"):
    st.write("Under construction")
