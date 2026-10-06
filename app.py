import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os

# Disable CUDA to prevent GPU segfaults on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from data_ingestion import fetch_realtime_air_quality_batch, INDIAN_CITIES
from anomaly_detection import EnvironmentalAnomalyDetector
from llm_agent import EnvironmentalAgent
import email_service
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Canopy EcoAI | Environmental Intelligence",
    layout="wide",
    page_icon="🍃",
    initial_sidebar_state="expanded"
)

# --- 1. Advanced Modern Custom CSS & Styling ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #f4f7f5;
        color: #111827;
    }

    /* Hide Streamlit Chrome but KEEP Sidebar Toggle */
    #MainMenu, footer {visibility: hidden;}
    header {background: transparent !important;}
    .stDeployButton {display: none;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b1d17 !important;
        background-image: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 50%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Content Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px;
    }

    /* Streamlit Selectbox Styling */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10b981 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1) !important;
    }

    /* Input Fields Styling */
    .stTextInput input {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 12px 16px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }
    .stTextInput input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
    }

    /* Primary & Standard Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
    }
    .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
    }
    .stButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar Glassmorphic Buttons */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #cbd5e1 !important;
        opacity: 0.75 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(16, 185, 129, 0.4) !important;
        color: #ffffff !important;
        opacity: 1 !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.3) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.6) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
        opacity: 1 !important;
    }

    /* Radio Buttons to Glassmorphism Pill Buttons Override */
    div[data-testid="stRadio"] > label {
        color: #94a3b8 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        gap: 8px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        padding: 5px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(8px) !important;
        opacity: 0.75 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        opacity: 1 !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.3) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.6) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
    }

    /* Streamlit Alert Boxes */
    .stAlert {
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
    }

    /* Custom Glass Cards */
    .eco-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .eco-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.08);
    }

    /* Simulator Panel Container */
    .simulator-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-top: 1.5rem;
        backdrop-filter: blur(12px);
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Custom HTML Metric Card ---
def metric_card(title, value, unit, is_anomaly=False, icon="📊"):
    status_text = "Critical Spike" if is_anomaly else "Stable Baseline"
    status_bg = "rgba(239, 68, 68, 0.12)" if is_anomaly else "rgba(16, 185, 129, 0.12)"
    status_color = "#dc2626" if is_anomaly else "#059669"
    dot_color = "#ef4444" if is_anomaly else "#10b981"
    card_border = "1px solid rgba(239, 68, 68, 0.3)" if is_anomaly else "1px solid rgba(226, 232, 240, 0.8)"
    bg_gradient = "linear-gradient(145deg, #ffffff 0%, #fff5f5 100%)" if is_anomaly else "linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)"
    
    html = f"""
    <div style="
        background: {bg_gradient};
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        border: {card_border};
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="color: #64748b; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px;">{title}</span>
            <span style="font-size: 18px; opacity: 0.8;">{icon}</span>
        </div>
        <div style="color: #0f172a; font-size: 36px; font-weight: 800; line-height: 1.1; letter-spacing: -0.5px;">
            {value} <span style="font-size: 14px; font-weight: 600; color: #64748b;">{unit}</span>
        </div>
        <div style="margin-top: 16px; display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 20px; background: {status_bg};">
            <div style="width: 7px; height: 7px; border-radius: 50%; background-color: {dot_color}; margin-right: 7px;"></div>
            <span style="font-size: 12px; font-weight: 700; color: {status_color};">{status_text}</span>
        </div>
    </div>
    """
    return html

@st.cache_resource(show_spinner=False)
def load_models_v2():
    detector = EnvironmentalAnomalyDetector()
    detector.load()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    llm_agent = EnvironmentalAgent(gemini_api_key=api_key)
    return detector, llm_agent

detector, llm_agent = load_models_v2()

# Initialize session state for AI Provider
if 'ai_provider' not in st.session_state:
    st.session_state.ai_provider = "Gemini (Google)"

# --- 3. Sidebar Layout ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <div style="display: inline-flex; align-items: center; justify-content: center; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 14px; width: 48px; height: 48px; margin-bottom: 12px;">
        <span style="font-size: 24px;">🍃</span>
    </div>
    <h2 style="color: #ffffff; font-weight: 800; font-size: 22px; margin: 0; letter-spacing: -0.5px;">Canopy EcoAI</h2>
    <p style="color: #94a3b8; font-size: 12px; margin-top: 4px; font-weight: 500;">Environmental Intelligence System</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    route = option_menu(
        menu_title=None,
        options=["Overview Dashboard", "National Map", "Alerts & Settings"],
        icons=["grid-1x2-fill", "geo-alt-fill", "bell-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "4px!important", "background-color": "rgba(255,255,255,0.03)!important", "border-radius": "16px", "border": "1px solid rgba(255,255,255,0.06)"},
            "icon": {"color": "#10b981", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "4px 0px",
                "padding": "10px 16px", 
                "--hover-color": "rgba(255,255,255,0.06)", 
                "color": "#cbd5e1",
                "border-radius": "10px",
                "font-weight": "600",
                "transition": "all 0.2s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%)",
                "color": "#ffffff",
                "border": "1px solid rgba(16, 185, 129, 0.4)",
                "font-weight": "700"
            }
        }
    )

st.sidebar.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.1); margin: 25px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 12px;">
    <span style="font-size: 16px; margin-right: 8px;">⚙️</span>
    <span style="color: #ffffff; font-weight: 700; font-size: 14px;">Simulator Control</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    demo_mode = st.toggle("Inject Demo Anomaly", key="demo_mode_toggle")
    demo_city = st.selectbox("Target Region", list(INDIAN_CITIES.keys()), key="demo_city_select") if demo_mode else None

st.sidebar.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.1); margin: 25px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 12px;">
    <span style="font-size: 16px; margin-right: 8px;">🧠</span>
    <span style="color: #ffffff; font-weight: 700; font-size: 14px;">AI Engine Settings</span>
</div>
<div style="font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">ACTIVE MODEL PROVIDER</div>
""", unsafe_allow_html=True)

with st.sidebar:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        is_gemini = (st.session_state.ai_provider == "Gemini (Google)")
        if st.button("✨ Gemini", key="btn_provider_gemini", use_container_width=True, type="primary" if is_gemini else "secondary"):
            st.session_state.ai_provider = "Gemini (Google)"
            st.rerun()
    with btn_col2:
        is_groq = (st.session_state.ai_provider == "Llama 3 (Groq)")
        if st.button("⚡ Groq", key="btn_provider_groq", use_container_width=True, type="primary" if is_groq else "secondary"):
            st.session_state.ai_provider = "Llama 3 (Groq)"
            st.rerun()
            
    ai_provider = st.session_state.ai_provider
    groq_key = os.environ.get("GROQ_API_KEY", "")


# --- 4. Fetch & Process Data ---
live_data_dict = fetch_realtime_air_quality_batch()
if demo_mode and demo_city in live_data_dict:
    live_data_dict[demo_city]['pm2_5'] = 195.0
    live_data_dict[demo_city]['carbon_monoxide'] = 980.0

city_metrics = []
for city, data in live_data_dict.items():
    is_anom, score = detector.detect_anomaly(data)
    data['anomaly_score'] = score
    data['is_anomaly'] = is_anom
    data['city'] = city
    city_metrics.append(data)
df = pd.DataFrame(city_metrics)

# --- 5. Routes ---

if route == "Overview Dashboard":
    # Header Banner
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <h1 style="color: #0f172a; font-weight: 800; font-size: 28px; margin: 0; letter-spacing: -0.5px;">Environmental Overview</h1>
            <span style="background: rgba(16, 185, 129, 0.12); color: #059669; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(16, 185, 129, 0.2);">LIVE MONITORING</span>
        </div>
        <p style="color: #64748b; font-size: 15px; margin-top: 6px; font-weight: 500;">Real-time environmental intelligence, sensor telemetries, and AI anomaly detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Region Selection Row
    col_sel, col_space = st.columns([1, 2])
    with col_sel:
        selected_city = st.selectbox("Select Active Monitoring Region", list(INDIAN_CITIES.keys()), index=0)
    
    city_data = live_data_dict[selected_city]
    is_anom = city_data['is_anomaly']
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # Custom HTML Metric Cards Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("PM 2.5 INDEX", city_data['pm2_5'], "µg/m³", is_anom, icon="🌫️"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("CARBON MONOXIDE", city_data['carbon_monoxide'], "µg/m³", is_anom, icon="💨"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("PM 10 LEVEL", city_data['pm10'], "µg/m³", False, icon="🍃"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("NITROGEN DIOXIDE", city_data['nitrogen_dioxide'], "µg/m³", False, icon="🧪"), unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Lower Section
    col_chart, col_ai = st.columns([1.4, 1])
    
    with col_chart:
        st.markdown("""
        <div class="eco-card" style="min-height: 440px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #0f172a; font-weight: 700; font-size: 18px; margin: 0;">Pollutant Radar Analysis</h3>
                <span style="color: #94a3b8; font-size: 13px; font-weight: 500;">Multi-axis metrics</span>
            </div>
        """, unsafe_allow_html=True)
        
        categories = ['PM2.5', 'PM10', 'CO (scaled)', 'NO2', 'Ozone']
        values = [city_data['pm2_5'], city_data['pm10'], city_data['carbon_monoxide']/10, city_data['nitrogen_dioxide'], city_data['ozone']]
        
        radar_color = '#ef4444' if is_anom else '#10b981'
        radar_fill = 'rgba(239, 68, 68, 0.25)' if is_anom else 'rgba(16, 185, 129, 0.2)'
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=values, theta=categories, fill='toself',
          line=dict(color=radar_color, width=3),
          fillcolor=radar_fill,
          marker=dict(size=6, color=radar_color)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=False, gridcolor='rgba(226, 232, 240, 0.8)'),
                angularaxis=dict(gridcolor='rgba(226, 232, 240, 0.8)', tickfont=dict(size=12, color='#475569', family='Plus Jakarta Sans'))
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ai:
        st.markdown("""
        <div class="eco-card" style="min-height: 440px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #0f172a; font-weight: 700; font-size: 18px; margin: 0;">AI Intelligence Report</h3>
                <span style="background: rgba(99, 102, 241, 0.1); color: #6366f1; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px;">{ai_provider.upper()} AGENT</span>
            </div>
        """, unsafe_allow_html=True)
        
        if is_anom:
            with st.spinner("Analyzing threat telemetry..."):
                provider_str = "Groq" if "Groq" in ai_provider else "Gemini"
                report = llm_agent.analyze_anomaly(city_data, city_data['anomaly_score'], provider=provider_str, groq_key=groq_key)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%); border: 1px solid rgba(239, 68, 68, 0.3); border-left: 5px solid #ef4444; padding: 20px; border-radius: 14px; color: #991b1b; font-size: 14px; line-height: 1.6;">
                    <div style="font-weight: 800; font-size: 16px; margin-bottom: 8px; color: #991b1b; display: flex; align-items: center;">
                        <span style="font-size: 20px; margin-right: 8px;">🚨</span> Threat Alert (Score: {city_data['anomaly_score']:.2f})
                    </div>
                    {report}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%); border: 1px solid rgba(16, 185, 129, 0.3); border-left: 5px solid #10b981; padding: 20px; border-radius: 14px; color: #166534; font-size: 14px; line-height: 1.6;">
                <div style="font-weight: 800; font-size: 16px; margin-bottom: 8px; color: #166534; display: flex; align-items: center;">
                    <span style="font-size: 20px; margin-right: 8px;">✅</span> Baseline Nominal
                </div>
                Air quality metrics are currently tracking along expected baseline parameters. No anomalous combustion or pollutant events detected in the selected region.
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

elif route == "National Map":
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <h1 style="color: #0f172a; font-weight: 800; font-size: 28px; margin: 0; letter-spacing: -0.5px;">Global Risk Map</h1>
            <span style="background: rgba(14, 165, 233, 0.12); color: #0284c7; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(14, 165, 233, 0.2);">GEOSPATIAL TELEMETRY</span>
        </div>
        <p style="color: #64748b; font-size: 15px; margin-top: 6px; font-weight: 500;">Live geographic tracking of critical environmental risk zones across India.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='eco-card' style='padding: 16px;'>", unsafe_allow_html=True)
    fig_map = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="anomaly_score", size="pm2_5",
        color_continuous_scale=[(0, "#10b981"), (0.5, "#f59e0b"), (1.0, "#ef4444")],
        range_color=[0, detector.threshold * 1.5],
        hover_name="city", mapbox_style="carto-positron",
        zoom=4, center={"lat": 22.0, "lon": 79.0}
    )
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        height=620,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif route == "Alerts & Settings":
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <h1 style="color: #0f172a; font-weight: 800; font-size: 28px; margin: 0; letter-spacing: -0.5px;">Notification Center</h1>
            <span style="background: rgba(99, 102, 241, 0.12); color: #4f46e5; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(99, 102, 241, 0.2);">ALERT REGISTRY</span>
        </div>
        <p style="color: #64748b; font-size: 15px; margin-top: 6px; font-weight: 500;">Configure automated notifications and dispatch triggers for detected anomalies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("""
            <div class="eco-card">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    <div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(16, 185, 129, 0.12); display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                        <span style="font-size: 20px;">📩</span>
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #0f172a; font-weight: 700; font-size: 18px;">Subscribe to Alerts</h3>
                        <p style="color: #64748b; font-size: 13px; margin: 2px 0 0 0;">Receive instant AI diagnostic reports on severe anomalies.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            email_sub = st.text_input("Email Address", key="sub", placeholder="operator@environmental-ai.org")
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Subscribe Now", type="primary", use_container_width=True):
                if email_service.add_subscriber(email_sub):
                    st.success(f"Successfully registered: {email_sub}")
                else:
                    st.error("Invalid email address or subscriber already registered.")
            
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown("""
            <div class="eco-card">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    <div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(239, 68, 68, 0.12); display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                        <span style="font-size: 20px;">🔕</span>
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #0f172a; font-weight: 700; font-size: 18px;">Manage Subscription</h3>
                        <p style="color: #64748b; font-size: 13px; margin: 2px 0 0 0;">Remove an existing email address from automated alerts.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            email_unsub = st.text_input("Registered Email Address", key="unsub", placeholder="operator@environmental-ai.org")
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Unsubscribe", type="secondary", use_container_width=True):
                if email_service.remove_subscriber(email_unsub):
                    st.success(f"Successfully unsubscribed: {email_unsub}")
                else:
                    st.error("Email address not found in registry.")
            
            st.markdown("</div>", unsafe_allow_html=True)
