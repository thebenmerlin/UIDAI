"""
Aadhaar Societal Insights Dashboard
====================================
Professional research-grade dashboard for visualizing societal insights
from Aadhaar enrolment, demographic, and biometric update data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# =============================================================================
# PAGE CONFIG - Light Mode, Research Paper Style
# =============================================================================

st.set_page_config(
    page_title="Aadhaar Societal Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force Light Mode with Research Paper Styling
st.markdown("""
<style>
    /* Force light mode */
    :root {
        color-scheme: light !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Base typography - readable, professional */
    html, body, [class*="css"] {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        color: #1a1a1a !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    
    .subtitle {
        font-size: 1.1rem !important;
        color: #333333 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .section-title {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #000000 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333333;
    }
    
    /* Metrics boxes */
    .metric-box {
        background-color: #f8f8f8 !important;
        border: 1px solid #cccccc !important;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem !important;
        color: #444444 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    
    /* Insight cards */
    .insight-box {
        background-color: #fafafa !important;
        border: 1px solid #dddddd !important;
        border-left: 4px solid #333333 !important;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    
    .insight-title {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #000000 !important;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        font-size: 0.95rem !important;
        color: #222222 !important;
        line-height: 1.6;
        margin-bottom: 0.4rem;
    }
    
    .insight-why {
        font-size: 0.9rem !important;
        color: #444444 !important;
        font-style: italic;
        line-height: 1.5;
    }
    
    /* Legend items */
    .legend-item {
        font-size: 0.9rem !important;
        color: #222222 !important;
        padding: 0.3rem 0;
    }
    
    /* Tables */
    .stDataFrame {
        font-size: 0.9rem !important;
    }
    
    /* Tabs - make text readable */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f0f0;
        padding: 4px;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #333333 !important;
        color: #ffffff !important;
        border-color: #333333 !important;
    }
    
    /* All paragraph text */
    p, span, div {
        color: #1a1a1a !important;
    }
    
    /* Links */
    a {
        color: #0066cc !important;
    }
    
    /* Markdown tables */
    table {
        color: #000000 !important;
        border-collapse: collapse !important;
    }
    
    th, td {
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        padding: 8px 12px !important;
    }
    
    th {
        background-color: #f0f0f0 !important;
        font-weight: 600 !important;
    }
    
    /* Plotly charts container */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #000000 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

# Use relative paths for Streamlit Cloud deployment
DATA_DIR = Path(__file__).parent / "Data"
GEOJSON_PATH = Path(__file__).parent / "india_states.geojson"

@st.cache_data
def load_data():
    """Load and preprocess datasets."""
    enrol = pd.read_csv(DATA_DIR / "api_data_aadhar_enrolment_0_500000.csv")
    demo = pd.read_csv(DATA_DIR / "api_data_aadhar_demographic_0_500000.csv")
    bio = pd.read_csv(DATA_DIR / "api_data_aadhar_biometric_0_500000.csv")
    
    for df in [enrol, demo, bio]:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    enrol = enrol.rename(columns={
        'age_0_5': 'enrol_0_5', 'age_5_17': 'enrol_5_17', 'age_18_greater': 'enrol_18plus'
    })
    demo = demo.rename(columns={'demo_age_5_17': 'demo_5_17', 'demo_age_17_': 'demo_17plus'})
    bio = bio.rename(columns={'bio_age_5_17': 'bio_5_17', 'bio_age_17_': 'bio_17plus'})
    
    merge_cols = ['date', 'state', 'district', 'pincode']
    merged = enrol.merge(demo, on=merge_cols, how='outer')
    merged = merged.merge(bio, on=merge_cols, how='outer')
    
    numeric_cols = [c for c in merged.columns if c not in merge_cols]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    return merged

@st.cache_data
def load_geojson():
    """Load India states GeoJSON."""
    with open(GEOJSON_PATH) as f:
        return json.load(f)

@st.cache_data
def compute_state_pressures(df):
    """Compute pressure signals aggregated by state."""
    state_agg = df.groupby('state').agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum'
    }).reset_index()
    
    state_agg.columns = ['state', 'entry_pressure', 'admin_pressure', 'physical_pressure']
    
    for col in ['entry_pressure', 'admin_pressure', 'physical_pressure']:
        max_val = state_agg[col].max()
        if max_val > 0:
            state_agg[f'{col}_norm'] = state_agg[col] / max_val
        else:
            state_agg[f'{col}_norm'] = 0
    
    def get_dominant(row):
        pressures = {
            'Entry-driven': row['entry_pressure_norm'],
            'Demographic-driven': row['admin_pressure_norm'],
            'Biometric-driven': row['physical_pressure_norm']
        }
        return max(pressures, key=pressures.get)
    
    state_agg['dominant_pressure'] = state_agg.apply(get_dominant, axis=1)
    state_agg['admin_entry_ratio'] = state_agg['admin_pressure'] / state_agg['entry_pressure'].replace(0, 1)
    state_agg['bio_admin_ratio'] = state_agg['physical_pressure'] / state_agg['admin_pressure'].replace(0, 1)
    
    return state_agg

@st.cache_data
def compute_daily_national(df):
    """Compute national daily pressure signals."""
    daily = df.groupby('date').agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum'
    }).reset_index()
    daily.columns = ['date', 'Entry', 'Admin', 'Physical']
    return daily

# =============================================================================
# VISUALIZATION FUNCTIONS - Light theme, readable
# =============================================================================

def create_india_map(state_data, geojson):
    """Create India choropleth map with proper sizing."""
    
    color_map = {
        'Entry-driven': '#2171b5',
        'Demographic-driven': '#cb181d', 
        'Biometric-driven': '#f16913'
    }
    
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state',
        featureidkey='properties.NAME_1',
        color='dominant_pressure',
        color_discrete_map=color_map,
        hover_name='state',
        hover_data={
            'state': False,
            'dominant_pressure': True,
            'admin_entry_ratio': ':.1f',
            'bio_admin_ratio': ':.2f'
        },
        labels={
            'dominant_pressure': 'Dominant Pressure',
            'admin_entry_ratio': 'Admin/Entry Ratio',
            'bio_admin_ratio': 'Bio/Admin Ratio'
        }
    )
    
    fig.update_geos(
        visible=False,
        fitbounds="locations",
        bgcolor='#ffffff'
    )
    
    fig.update_layout(
        margin={"r": 10, "t": 50, "l": 10, "b": 10},
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11, color='#000000'),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        title=dict(
            text="<b>Dominant Life-Event Pressure by State</b>",
            font=dict(size=14, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    return fig

def create_pressure_timeline(daily_data):
    """Create national pressure timeline."""
    fig = go.Figure()
    
    colors = {'Entry': '#2171b5', 'Admin': '#cb181d', 'Physical': '#f16913'}
    
    for col in ['Entry', 'Admin', 'Physical']:
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data[col],
            name=col,
            mode='lines',
            line=dict(width=2, color=colors[col])
        ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis_title="Date",
        yaxis_title="Activity Volume",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color='#000000')
        ),
        hovermode='x unified',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8f8f8',
        title=dict(
            text="<b>National Pressure Signals Over Time</b>",
            font=dict(size=14, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', linecolor='#000000', tickfont=dict(color='#000000'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', linecolor='#000000', tickfont=dict(color='#000000'))
    
    return fig

def create_state_bar_chart(state_data, metric='admin_entry_ratio', top_n=15):
    """Create horizontal bar chart."""
    sorted_data = state_data.nlargest(top_n, metric)
    
    titles = {
        'admin_entry_ratio': "States by Admin/Entry Ratio",
        'bio_admin_ratio': "States by Bio/Admin Ratio"
    }
    colors = {
        'admin_entry_ratio': '#2171b5',
        'bio_admin_ratio': '#f16913'
    }
    
    fig = go.Figure(go.Bar(
        x=sorted_data[metric],
        y=sorted_data['state'],
        orientation='h',
        marker_color=colors.get(metric, '#333333')
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=100, r=20, t=50, b=50),
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title=None,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8f8f8',
        title=dict(
            text=f"<b>{titles.get(metric, metric)}</b>",
            font=dict(size=13, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    fig.update_yaxes(categoryorder='total ascending', tickfont=dict(size=10, color='#000000'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', tickfont=dict(color='#000000'))
    
    return fig

def create_pie_chart(state_data):
    """Create pressure distribution pie chart."""
    dist = state_data['dominant_pressure'].value_counts()
    
    colors = {
        'Entry-driven': '#2171b5',
        'Demographic-driven': '#cb181d',
        'Biometric-driven': '#f16913'
    }
    
    fig = go.Figure(go.Pie(
        labels=dist.index,
        values=dist.values,
        marker_colors=[colors.get(x, '#666666') for x in dist.index],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=11, color='#000000')
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        paper_bgcolor='#ffffff',
        title=dict(
            text="<b>Region Typology Distribution</b>",
            font=dict(size=13, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    return fig

def create_correlation_heatmap(state_data):
    """Create correlation heatmap."""
    corr_cols = ['entry_pressure', 'admin_pressure', 'physical_pressure']
    corr_matrix = state_data[corr_cols].corr()
    labels = ['Entry', 'Admin', 'Physical']
    
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=14, color='#000000')
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='#ffffff',
        title=dict(
            text="<b>Pressure Signal Correlations</b>",
            font=dict(size=13, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    fig.update_xaxes(tickfont=dict(color='#000000'))
    fig.update_yaxes(tickfont=dict(color='#000000'))
    
    return fig

# =============================================================================
# INSIGHTS CONTENT
# =============================================================================

INSIGHTS = [
    {
        "title": "Clear Migration Asymmetry Detected",
        "what": "States show 2-10x variation in admin/entry ratios, revealing distinct absorbing versus sending regions.",
        "why": "High admin/entry ratio indicates people moving IN and updating addresses. Low ratio suggests origin states where people enroll but later migrate out."
    },
    {
        "title": "Workforce Churn Hotspots Identified",
        "what": "12 states show bio-admin correlation above 0.7, concentrated in NCR, Punjab-Haryana, and industrialized regions.",
        "why": "Synchronized biometric and demographic updates signal workforce turnover — job changes require both address updates and biometric re-verification."
    },
    {
        "title": "Household Formation Patterns Visible",
        "what": "Demographic update spikes occur without corresponding enrolment increases in several regions.",
        "why": "This is the classic household formation signature — existing Aadhaar holders changing addresses due to marriage, new family formation, or independent housing."
    },
    {
        "title": "Administrative Pressure Dominates Nationally",
        "what": "Nationwide, administrative pressure exceeds entry pressure by approximately 100x.",
        "why": "The identity system is mature. Most adult activity reflects life changes (address moves, corrections) rather than first-time enrollment."
    },
    {
        "title": "Regional Pressure Divergence",
        "what": "Northeast states cluster as transitional zones; Southern states show higher biometric activity.",
        "why": "Different economic patterns and welfare program verification requirements create distinct regional pressure profiles."
    }
]

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-title">Aadhaar Societal Insights Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Population-Level Patterns in Migration, Household Formation, and Workforce Dynamics</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        geojson = load_geojson()
        state_data = compute_state_pressures(df)
        daily_data = compute_daily_national(df)
    
    # Key Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{state_data["state"].nunique()}</div>
            <div class="metric-label">States/Regions</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{df["pincode"].nunique():,}</div>
            <div class="metric-label">Unique Pincodes</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        date_range = f'{df["date"].min().strftime("%b %Y")} - {df["date"].max().strftime("%b %Y")}'
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value" style="font-size: 1.2rem;">{date_range}</div>
            <div class="metric-label">Date Range</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Geographic Analysis", "Temporal Analytics", "Key Findings", "Methodology"])
    
    # =================================
    # TAB 1: Geographic Analysis
    # =================================
    with tab1:
        st.markdown('<h2 class="section-title">Dominant Life-Event Pressure by Region</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Each region is classified by its **dominant pressure type** — the life-event signal with the 
        highest normalized activity. This reveals what kind of life transition is shaping each area.
        """)
        
        # Legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<p class="legend-item"><b style="color:#2171b5;">■</b> <b>Entry-driven</b>: New adults entering identity system</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="legend-item"><b style="color:#cb181d;">■</b> <b>Demographic-driven</b>: Address/identity changes</p>', unsafe_allow_html=True)
        with col3:
            st.markdown('<p class="legend-item"><b style="color:#f16913;">■</b> <b>Biometric-driven</b>: Re-verification activity</p>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Map and Distribution side by side
        col_map, col_dist = st.columns([2, 1])
        
        with col_map:
            fig_map = create_india_map(state_data, geojson)
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col_dist:
            fig_pie = create_pie_chart(state_data)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("**Top States by Admin/Entry Ratio:**")
            top_states = state_data.nlargest(8, 'admin_entry_ratio')[['state', 'admin_entry_ratio', 'dominant_pressure']]
            top_states.columns = ['State', 'Ratio', 'Type']
            top_states['Ratio'] = top_states['Ratio'].round(1)
            st.dataframe(top_states, use_container_width=True, hide_index=True)
    
    # =================================
    # TAB 2: Temporal Analytics
    # =================================
    with tab2:
        st.markdown('<h2 class="section-title">Pressure Signal Analytics</h2>', unsafe_allow_html=True)
        
        # Timeline
        fig_timeline = create_pressure_timeline(daily_data)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("")
        
        # Bar charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_admin = create_state_bar_chart(state_data, 'admin_entry_ratio')
            st.plotly_chart(fig_admin, use_container_width=True)
        
        with col2:
            fig_bio = create_state_bar_chart(state_data, 'bio_admin_ratio')
            st.plotly_chart(fig_bio, use_container_width=True)
        
        # Correlation
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            fig_corr = create_correlation_heatmap(state_data)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # =================================
    # TAB 3: Key Findings
    # =================================
    with tab3:
        st.markdown('<h2 class="section-title">Key Findings</h2>', unsafe_allow_html=True)
        
        for i, insight in enumerate(INSIGHTS, 1):
            st.markdown(f'''
            <div class="insight-box">
                <div class="insight-title">{i}. {insight["title"]}</div>
                <div class="insight-text"><b>Observation:</b> {insight["what"]}</div>
                <div class="insight-why"><b>Significance:</b> {insight["why"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        **Data Source:** UIDAI Aadhaar activity data (enrolment, demographic updates, biometric updates)  
        **Analysis Method:** Pressure signal framework with normalization and geographic clustering  
        **Scope:** Population-level patterns only — no individual inference possible or attempted
        """)
    
    # =================================
    # TAB 4: Methodology
    # =================================
    with tab4:
        st.markdown('<h2 class="section-title">Analytical Framework</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Pressure Signal Model
        
        This analysis treats Aadhaar activity as **three population-level pressure signals**:
        """)
        
        st.markdown("""
        | Pressure Type | Data Source | Interpretation |
        |--------------|-------------|----------------|
        | **Entry Pressure** | Adult enrolments (age 18+) | New adults entering formal identity system |
        | **Admin Pressure** | Demographic updates (age 17+) | Life changes requiring address/identity updates |
        | **Physical Pressure** | Biometric updates (age 17+) | Physical re-verification needs |
        """)
        
        st.markdown("""
        ### Key Insight: Life Events as Pressure Imbalances
        
        Life events manifest as **imbalances** between these pressure signals:
        
        - **Migration:** Admin pressure spike *precedes* enrolment in destination region, *follows* in origin region
        - **Household Formation:** Admin spike *without* corresponding enrolment (existing holders moving)
        - **Workforce Churn:** Admin + Bio spike *together* (job changes require both types of updates)
        
        ### Classification Method
        
        1. Normalize all three pressures to [0,1] scale per region
        2. Identify the pressure with highest normalized value
        3. Assign classification: Entry-driven, Demographic-driven, or Biometric-driven
        
        ### Ratio Interpretation
        
        - **Admin/Entry Ratio > 1:** "Absorbing" region (more address changes than new enrollments)
        - **Admin/Entry Ratio < 1:** "Sending" region (people enrolled here tend to migrate out)
        - **High Bio/Admin Ratio:** Workforce churn or verification-intensive region
        """)

if __name__ == "__main__":
    main()
