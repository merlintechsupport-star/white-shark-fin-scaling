import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64

# Custom CSS for ocean background
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3e7cb1 100%);
    background-attachment: fixed;
}
.stApp {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 20px;
    margin: 20px;
}
.sidebar .sidebar-content {
    background: rgba(0, 123, 255, 0.1);
}
h1, h2, h3 {
    color: #2c3e50;
}
.metric-card {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="manage-app-button"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv('data/merged_shark_data.csv')
df = df.dropna()

# Compute fits
log_TL = np.log(df['TL_cm'])
log_pec = np.log(df['avg_pec_cm2'])
coeffs = np.polyfit(log_TL, log_pec, 1)
b_ols = coeffs[0]
intercept_ols = coeffs[1]
a_ols = np.exp(intercept_ols)
r = np.corrcoef(log_TL, log_pec)[0,1]
b_sma = b_ols / r
mean_log_TL = np.mean(log_TL)
mean_log_pec = np.mean(log_pec)
a_sma = np.exp(mean_log_pec - b_sma * mean_log_TL)

# Page config
st.set_page_config(page_title="Stanford Hopkins White Shark Fin Scaling", page_icon="🦈", layout="wide")

# Sidebar
st.sidebar.title("🦈 Stanford Hopkins Marine Station")
st.sidebar.markdown("**White Shark Pectoral Fin Scaling Project**")

st.sidebar.markdown("**Created by:** Justin Yu")
st.sidebar.markdown("**Email:** justinyu@stanford.edu")

page = st.sidebar.radio("Navigate", ["Home", "Data Visualization", "Findings & Discussion", "Project Pipeline"])

# Home
if page == "Home":
    st.title("White Shark Pectoral Fin Scaling Analysis")
    st.markdown("### Stanford Hopkins Marine Station")
    
    st.markdown("""
    Hey there! I'm Justin, and I put this together for Alexandra's white shark research project. This is a super rough draft, there's definitely more work to be done, but I wanted to get the pipeline running and show what it could look like.
    
    **Quick disclaimer:** Since I didn't have the API key to pull the real CSV and JSON files from Labelbox, I just made up some random data to test the whole pipeline. So yeah, these results are totally fake for now!
    
    This dashboard dives into whether pectoral fin area scales proportionally with body size in eastern Pacific white sharks, compared to Australian ones where the fins kinda fall behind in bigger sharks.
    
    **Big questions we're tackling:**
    - Do eastern Pacific white sharks have isometric scaling (b=2, meaning fins grow in proportion)?
    - Or do they show negative allometry like the Aussies (b<2, fins get relatively smaller)?
    
    Poke around the sidebar to check out the data, what we found, and how we did it.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Sharks", len(df))
        st.metric("Size Range", f"{df['TL_cm'].min():.0f} - {df['TL_cm'].max():.0f} cm")
    with col2:
        st.metric("OLS Slope (b)", f"{b_ols:.3f}")
        st.metric("SMA Slope (b)", f"{b_sma:.3f}")

# Data Visualization
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.markdown("**⚠️ HEADS UP: This is all simulated data, not real measurements!**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Fitted Models", "Residuals", "Interactive Plot"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8,6))
            colors = {'M': '#3498db', 'F': '#e74c3c'}  # Cooler blues and reds
            for sex in df['sex'].unique():
                subset = df[df['sex'] == sex]
                ax.scatter(subset['TL_cm'], subset['avg_pec_cm2'], color=colors[sex], label=f'{sex}ales', alpha=0.8, s=60, edgecolors='white')
            ax.set_xlabel('Total Length (cm)', fontsize=12)
            ax.set_ylabel('Avg Pectoral Fin Area (cm²)', fontsize=12)
            ax.set_title('Total Length vs Avg Pectoral Fin Area', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(df['left_pec_cm2'], df['right_pec_cm2'], alpha=0.8, s=60, edgecolors='white', color='#27ae60')
            min_val = min(df['left_pec_cm2'].min(), df['right_pec_cm2'].min())
            max_val = max(df['left_pec_cm2'].max(), df['right_pec_cm2'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line', linewidth=2)
            ax.set_xlabel('Left Pectoral Fin Area (cm²)', fontsize=12)
            ax.set_ylabel('Right Pectoral Fin Area (cm²)', fontsize=12)
            ax.set_title('Left vs Right Fin Symmetry', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10,8))
        colors = {'M': '#3498db', 'F': '#e74c3c'}
        for sex in df['sex'].unique():
            subset = df[df['sex'] == sex]
            ax.scatter(subset['TL_cm'], subset['avg_pec_cm2'], color=colors[sex], label=f'{sex}ales', alpha=0.8, s=60, edgecolors='white')
        
        TL_range = np.linspace(df['TL_cm'].min(), df['TL_cm'].max(), 100)
        ax.plot(TL_range, a_ols * TL_range**b_ols, 'k-', label=f'OLS (b={b_ols:.3f})', linewidth=3)
        ax.plot(TL_range, a_sma * TL_range**b_sma, 'b--', label=f'SMA (b={b_sma:.3f})', linewidth=3)
        ax.plot(TL_range, a_ols * TL_range**b_ols, 'g:', label=f'Nonlinear (b={b_ols:.3f})', linewidth=3)
        ax.plot(TL_range, a_sma * TL_range**1.68, color='gray', linestyle='--', label='Literature (b=1.68)', linewidth=2)
        ax.set_xlabel('Total Length (cm)', fontsize=12)
        ax.set_ylabel('Avg Pectoral Fin Area (cm²)', fontsize=12)
        ax.set_title('All Methods, Raw Scale', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            pred_log_pec_ols = intercept_ols + b_ols * log_TL
            residuals_ols = log_pec - pred_log_pec_ols
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(df['TL_cm'], residuals_ols, alpha=0.8, s=60, edgecolors='white', color='#f39c12')
            ax.axhline(0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Total Length (cm)', fontsize=12)
            ax.set_ylabel('OLS Residuals (log scale)', fontsize=12)
            ax.set_title('OLS Residuals', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig)
            
        with col2:
            pred_log_pec_sma = np.log(a_sma) + b_sma * log_TL
            residuals_sma = log_pec - pred_log_pec_sma
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(df['TL_cm'], residuals_sma, alpha=0.8, s=60, edgecolors='white', color='#9b59b6')
            ax.axhline(0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Total Length (cm)', fontsize=12)
            ax.set_ylabel('SMA Residuals (log scale)', fontsize=12)
            ax.set_title('SMA Residuals', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Interactive Scatter Plot")
        x_axis = st.selectbox("X-axis", ["TL_cm", "log_TL"])
        y_axis = st.selectbox("Y-axis", ["avg_pec_cm2", "log_pec"])
        color_by = st.selectbox("Color by", ["sex", "none"])
        
        fig, ax = plt.subplots(figsize=(10,8))
        if color_by == "sex":
            colors = {'M': '#3498db', 'F': '#e74c3c'}
            for sex in df['sex'].unique():
                subset = df[df['sex'] == sex]
                x_data = np.log(subset['TL_cm']) if x_axis == "log_TL" else subset['TL_cm']
                y_data = np.log(subset['avg_pec_cm2']) if y_axis == "log_pec" else subset['avg_pec_cm2']
                ax.scatter(x_data, y_data, color=colors[sex], label=f'{sex}ales', alpha=0.8, s=60, edgecolors='white')
            ax.legend()
        else:
            x_data = np.log(df['TL_cm']) if x_axis == "log_TL" else df['TL_cm']
            y_data = np.log(df['avg_pec_cm2']) if y_axis == "log_pec" else df['avg_pec_cm2']
            ax.scatter(x_data, y_data, alpha=0.8, s=60, edgecolors='white', color='#27ae60')
        
        ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{x_axis} vs {y_axis}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

# Findings & Discussion
elif page == "Findings & Discussion":
    st.title("Findings & Discussion")
    st.markdown("**⚠️ IMPORTANT: All this is based on made-up data. Real analysis is coming once we get that API key!**")
    
    st.header("Key Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("OLS Slope (b)", f"{b_ols:.3f}")
        st.markdown("**What it means:** Negative allometry")
    with col2:
        st.metric("SMA Slope (b)", f"{b_sma:.3f}")
        st.markdown("**What it means:** Negative allometry")
    with col3:
        st.metric("Literature (Australian)", "1.68")
        st.markdown("**What it means:** Negative allometry")
    
    st.header("So What Does This Actually Mean?")
    st.markdown("""
    Alright, let's break down what these scaling relationships are all about:
    
    - **Isometric (b=2):** Fins grow right in proportion with the shark's body size, like if you doubled the shark's length, the fins double too.
    - **Positive allometry (b>2):** Fins grow disproportionately bigger, they outpace body growth.
    - **Negative allometry (b<2):** Fins grow disproportionately smaller, they lag behind as the shark gets bigger.
    
    **Our fake data shows:**
    - Both OLS and SMA give us b < 2, so negative allometry.
    - This means pectoral fins become relatively smaller as eastern Pacific white sharks grow larger.
    - Interestingly, this matches what they've seen in Australian white sharks (b=1.68).
    
    **Why might this matter?**
    - Could affect how efficiently they swim, maneuver, or even regulate their body temperature as adults.
    - Maybe tied to changes in where they live or how they hunt as they get bigger.
    - But hey, this is all speculative until we get real data!
    """)
    
    st.header("Data Limitations (And There's a Lot)")
    st.markdown("""
    - **Totally Simulated:** I just made this data up because I couldn't access the real CSV and JSON files without the API key.
    - **Tiny Sample:** Only 30 sharks (3 real ones, 27 fake).
    - **Limited Size Range:** Just 295-427 cm TL, not the full spectrum.
    - **Real Data Status:** Still waiting on Alexandra for that Labelbox API access.
    
    **What's Next:**
    1. Get the API key and pull actual fin outline data.
    2. Process the real pixel masks into true fin areas.
    3. Re-run everything with legit measurements.
    4. Compare eastern Pacific vs Australian populations properly.
    """)

# Project Pipeline
elif page == "Project Pipeline":
    st.title("Project Pipeline & Methodology")
    st.markdown("**Stanford Hopkins Marine Station, White Shark Research**")
    
    st.header("The Big Picture")
    st.markdown("""
    This whole project is about figuring out how pectoral fin size changes with body size in white sharks, using drone footage and some computer vision magic.
    The goal? Understand if fins scale the same way across different populations.
    """)
    
    st.header("How We Collect the Data")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Step 1: Drone Surveys")
        st.markdown("""
        - Fly drones over spots where white sharks hang out.
        - Snap high-res images.
        - Log GPS coords and how high we're flying.
        """)
        
        st.subheader("Step 2: Length Measurements")
        st.markdown("""
        - Use photogrammetry software to measure.
        - Get total length from nose to tail.
        - Calculate ground sampling distance (GSD) for scaling.
        """)
        
        st.subheader("Step 3: Fin Outlines")
        st.markdown("""
        - Manually draw polygons around the pectoral fins in Labelbox.
        - Creates pixel masks for each fin.
        """)
        
    with col2:
        st.subheader("Step 4: Data Processing")
        st.markdown("""
        - Pull the masks down via Labelbox API.
        - Count pixels in each fin mask.
        - Convert to cm² using GSD².
        - Average left and right fins.
        """)
        
        st.subheader("Step 5: Statistical Analysis")
        st.markdown("""
        - Log-transform length and area data.
        - Fit different regression models (OLS, SMA, nonlinear).
        - Compare against literature values.
        - Figure out the allometry patterns.
        """)
    
    st.header("The Nitty-Gritty Technical Stuff")
    with st.expander("Tools & Libraries We Use"):
        st.markdown("""
        - **Python Stack:** pandas for data wrangling, numpy for math, matplotlib for plots, scipy for stats.
        - **Data Processing:** Labelbox API for annotations, photogrammetry for measurements.
        - **Analysis:** Linear regression on logs, power law fitting.
        - **Visualization:** Matplotlib for static plots, interactive stuff here.
        """)
    
    with st.expander("Regression Methods Explained"):
        st.markdown("""
        **OLS (Ordinary Least Squares):**
        - The classic linear regression, but on log-transformed data.
        - The slope gives us the allometric exponent b.
        
        **SMA (Standardized Major Axis):**
        - Biologists love this for scaling studies.
        - Accounts for measurement error in both length and area.
        - Formula: b_SMA = OLS_slope / correlation_coefficient
        
        **Nonlinear Power Regression:**
        - Fits the power law directly: area = a × length^b
        - No need to log-transform first.
        """)
    
    st.header("Where the Data Comes From")
    st.markdown("""
    - **Eastern Pacific:** Drone surveys at aggregation sites (real data pending).
    - **Australian:** From papers like Kolborg et al. 2013 (b=1.72 OLS, b=1.68 SMA).
    - **Right Now:** Just testing with simulated data since no API key yet.
    """)
    
    st.header("Future Plans")
    st.markdown("""
    - Get way more sharks across all size ranges.
    - Compare multiple populations head-to-head.
    - Dig into what this means ecologically.
    - Maybe build automated fin detection to speed things up.
    """)

# Footer

st.markdown("*Stanford Hopkins Marine Station | White Shark Research Program*")
st.markdown("*Built by Justin Yu (justinyu@stanford.edu) for Alexandra DiGiacomo | Data: Simulated for demo*")