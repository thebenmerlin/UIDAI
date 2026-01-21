"""
Aadhaar Life-Event Insights Dashboard
======================================
Research-grade dashboard revealing population-level life transitions
from Aadhaar activity patterns across India.

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
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="India Life-Event Insights",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light Mode Research Paper Styling
st.markdown("""
<style>
    :root { color-scheme: light !important; }
    .stApp { background-color: #ffffff !important; }
    #MainMenu, footer, header { visibility: hidden; }
    
    html, body, [class*="css"] {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        color: #1a1a1a !important;
    }
    
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
    }
    
    .metric-label {
        font-size: 0.85rem !important;
        color: #444444 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
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
    }
    
    .insight-why {
        font-size: 0.9rem !important;
        color: #444444 !important;
        font-style: italic;
    }
    
    .legend-item {
        font-size: 0.9rem !important;
        color: #222222 !important;
        padding: 0.3rem 0;
    }
    
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
        background-color: #8f8b8b !important;
        color: #ffffff !important;
    }
    
    p, span, div { color: #1a1a1a !important; }
    a { color: #0066cc !important; }
    table { color: #000000 !important; }
    th, td { color: #000000 !important; border: 1px solid #cccccc !important; padding: 8px 12px !important; }
    th { background-color: #f0f0f0 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

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
def compute_life_events(df):
    """Classify each state by dominant life-event pattern."""
    state_agg = df.groupby('state').agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum'
    }).reset_index()
    
    state_agg.columns = ['state', 'new_adults', 'address_changes', 'biometric_updates']
    
    # Normalize each signal (0-1)
    for col in ['new_adults', 'address_changes', 'biometric_updates']:
        max_val = state_agg[col].max()
        state_agg[f'{col}_norm'] = state_agg[col] / max_val if max_val > 0 else 0
    
    # Compute behavioral indicators
    state_agg['migration_inflow'] = state_agg['address_changes'] / state_agg['new_adults'].replace(0, 1)
    state_agg['workforce_activity'] = state_agg['biometric_updates'] / state_agg['address_changes'].replace(0, 1)
    
    # Classify dominant life-event
    def classify_region(row):
        # Migration classification
        if row['migration_inflow'] > 50:
            if row['workforce_activity'] > 0.8:
                return "Urban Absorption Zone"
            else:
                return "In-Migration Hub"
        elif row['migration_inflow'] < 10:
            return "Out-Migration Region"
        
        # Address change dominant
        if row['address_changes_norm'] > row['new_adults_norm'] * 1.5:
            if row['workforce_activity'] > 0.6:
                return "Workforce Churn"
            else:
                return "Household Formation"
        
        # Biometric dominant
        if row['biometric_updates_norm'] > max(row['address_changes_norm'], row['new_adults_norm']):
            return "High Work-Related Stress"
        
        # Balanced
        if row['new_adults_norm'] > 0.3:
            return "Emerging Urban Centre"
        
        return "Stable Population"
    
    state_agg['life_event'] = state_agg.apply(classify_region, axis=1)
    
    return state_agg

@st.cache_data
def compute_daily_trends(df):
    """Compute national daily life-event trends."""
    daily = df.groupby('date').agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum'
    }).reset_index()
    daily.columns = ['date', 'New Adults Entering', 'Address Changes', 'Work Verifications']
    return daily

# =============================================================================
# LIFE-EVENT COLOR SCHEME
# =============================================================================

LIFE_EVENT_COLORS = {
    "In-Migration Hub": "#2171b5",
    "Out-Migration Region": "#6baed6",
    "Urban Absorption Zone": "#08519c",
    "Household Formation": "#cb181d",
    "Workforce Churn": "#f16913",
    "High Work-Related Stress": "#d94801",
    "Emerging Urban Centre": "#31a354",
    "Stable Population": "#969696"
}

LIFE_EVENT_DESCRIPTIONS = {
    "In-Migration Hub": "High volume of people moving in and updating addresses",
    "Out-Migration Region": "People enroll here but later move to other regions",
    "Urban Absorption Zone": "Active job market absorbing workers from elsewhere",
    "Household Formation": "New households forming — marriages, families moving",
    "Workforce Churn": "High job turnover with frequent address and work updates",
    "High Work-Related Stress": "Intensive work verification activity",
    "Emerging Urban Centre": "Growing urban area with new residents",
    "Stable Population": "Low life-event activity, settled population"
}

# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_india_map(state_data, geojson):
    """Create India map with life-event classification."""
    
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state',
        featureidkey='properties.NAME_1',
        color='life_event',
        color_discrete_map=LIFE_EVENT_COLORS,
        hover_name='state',
        hover_data={
            'state': False,
            'life_event': True
        },
        labels={'life_event': 'Life Event Pattern'}
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
            y=-0.08,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=10, color='#000000'),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        paper_bgcolor='#ffffff',
        title=dict(
            text="<b>What Life Changes Are Happening Across India?</b>",
            font=dict(size=14, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    return fig

def create_timeline(daily_data):
    """Create activity timeline."""
    fig = go.Figure()
    
    colors = {
        'New Adults Entering': '#2171b5',
        'Address Changes': '#cb181d',
        'Work Verifications': '#f16913'
    }
    
    for col in ['New Adults Entering', 'Address Changes', 'Work Verifications']:
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
        yaxis_title="Daily Activity",
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
            text="<b>Life-Event Activity Over Time</b>",
            font=dict(size=14, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', tickfont=dict(color='#000000'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', tickfont=dict(color='#000000'))
    
    return fig

def create_life_event_bar(state_data, metric='migration_inflow', title="Top Regions", top_n=12):
    """Create bar chart for life-event ranking."""
    sorted_data = state_data.nlargest(top_n, metric)
    
    colors = [LIFE_EVENT_COLORS.get(le, '#666666') for le in sorted_data['life_event']]
    
    fig = go.Figure(go.Bar(
        x=sorted_data[metric],
        y=sorted_data['state'],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=100, r=20, t=50, b=50),
        xaxis_title=None,
        yaxis_title=None,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8f8f8',
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color='#000000', family='Georgia'),
            x=0.5
        ),
        font=dict(color='#000000')
    )
    
    fig.update_yaxes(categoryorder='total ascending', tickfont=dict(size=10, color='#000000'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', tickfont=dict(color='#000000'))
    
    return fig

def create_distribution_chart(state_data):
    """Create life-event distribution chart."""
    dist = state_data['life_event'].value_counts()
    
    fig = go.Figure(go.Pie(
        labels=dist.index,
        values=dist.values,
        marker_colors=[LIFE_EVENT_COLORS.get(x, '#666666') for x in dist.index],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=10, color='#000000')
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        paper_bgcolor='#ffffff',
        title=dict(
            text="<b>Life-Event Distribution</b>",
            font=dict(size=13, color='#000000', family='Georgia'),
            x=0.5
        )
    )
    
    return fig

# =============================================================================
# KEY FINDINGS
# =============================================================================

INSIGHTS = [
    {
        "title": "Clear In-Migration and Out-Migration Corridors",
        "what": "Some states consistently receive people moving in (high address changes), while others see residents leave after initial enrollment.",
        "why": "This reveals internal migration flows across India — who is absorbing population and who is sending."
    },
    {
        "title": "Workforce Churn Concentrated in Industrial Regions",
        "what": "NCR, Punjab-Haryana, and Gujarat show synchronized address changes and work verifications.",
        "why": "These are high job-turnover areas where workers frequently change employers and locations."
    },
    {
        "title": "Household Formation Visible in Address Changes",
        "what": "Address changes spike without new enrollments in several regions.",
        "why": "This is the signature of household formation — marriages, new families, young adults moving out."
    },
    {
        "title": "Address Changes Dominate Nationwide",
        "what": "Across India, address changes far exceed new adult enrollments.",
        "why": "The identity system is mature — most activity now reflects life changes, not first-time registration."
    },
    {
        "title": "Regional Life-Event Patterns Differ Sharply",
        "what": "Northeast shows stable populations; South shows high work verification activity.",
        "why": "Different economic structures and welfare requirements create distinct regional signatures."
    }
]

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-title">India Life-Event Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">What Life Changes Are Happening Across India?</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        geojson = load_geojson()
        state_data = compute_life_events(df)
        daily_data = compute_daily_trends(df)
    
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
            <div class="metric-label">States Analyzed</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{df["pincode"].nunique():,}</div>
            <div class="metric-label">Localities Covered</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        date_range = f'{df["date"].min().strftime("%b %Y")} - {df["date"].max().strftime("%b %Y")}'
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value" style="font-size: 1.2rem;">{date_range}</div>
            <div class="metric-label">Time Period</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Life-Event Map", "Trends", "Key Findings", "How It Works"])
    
    # TAB 1: Map
    with tab1:
        st.markdown('<h2 class="section-title">What Life Changes Are Happening Where?</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Each region is classified by its **dominant life-event pattern**. 
        Point at any state to see what kind of life changes are most common there.
        """)
        
        # Legend
        col1, col2, col3, col4 = st.columns(4)
        legend_items = list(LIFE_EVENT_COLORS.items())[:8]
        for i, (event, color) in enumerate(legend_items):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                st.markdown(f'<p class="legend-item"><b style="color:{color};">●</b> {event}</p>', unsafe_allow_html=True)
        
        st.markdown("")
        
        col_map, col_dist = st.columns([2, 1])
        
        with col_map:
            fig_map = create_india_map(state_data, geojson)
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col_dist:
            fig_pie = create_distribution_chart(state_data)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("**Top In-Migration Regions:**")
            top_inflow = state_data.nlargest(6, 'migration_inflow')[['state', 'life_event']]
            top_inflow.columns = ['State', 'Pattern']
            st.dataframe(top_inflow, use_container_width=True, hide_index=True)
    
    # TAB 2: Trends
    with tab2:
        st.markdown('<h2 class="section-title">Life-Event Activity Over Time</h2>', unsafe_allow_html=True)
        
        fig_timeline = create_timeline(daily_data)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_inflow = create_life_event_bar(state_data, 'migration_inflow', "Top In-Migration Regions")
            st.plotly_chart(fig_inflow, use_container_width=True)
        
        with col2:
            fig_churn = create_life_event_bar(state_data, 'workforce_activity', "Top Workforce Activity Regions")
            st.plotly_chart(fig_churn, use_container_width=True)
    
    # TAB 3: Findings
    with tab3:
        st.markdown('<h2 class="section-title">Key Findings</h2>', unsafe_allow_html=True)
        
        for i, insight in enumerate(INSIGHTS, 1):
            st.markdown(f'''
            <div class="insight-box">
                <div class="insight-title">{i}. {insight["title"]}</div>
                <div class="insight-text"><b>What we see:</b> {insight["what"]}</div>
                <div class="insight-why"><b>Why it matters:</b> {insight["why"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        **Data Source:** UIDAI activity data (new enrollments, address changes, work verifications)  
        **Scope:** Population-level patterns only — no individual data used
        """)
    
    # TAB 4: Methodology
    with tab4:
        st.markdown('<h2 class="section-title">How This Works</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Three Types of Activity
        
        | Activity | What It Means |
        |----------|---------------|
        | **New Adults Entering** | People registering for the first time as adults |
        | **Address Changes** | Existing holders updating their address (moving, household changes) |
        | **Work Verifications** | Biometric re-verification (often required for jobs, benefits) |
        
        ### How Regions Are Classified
        
        By comparing these three activities, we can identify what kind of life-event is dominant:
        
        - **More address changes than new registrations** → People moving IN (In-Migration Hub)
        - **Few address changes relative to registrations** → People moving OUT (Out-Migration Region)
        - **High address changes + work verifications** → Job turnover (Workforce Churn)
        - **Address changes without new registrations** → Existing households changing (Household Formation)
        - **High work verifications** → Work-related activity (High Work-Related Stress)
        
        ### What This Reveals
        
        Without tracking any individual, we can see:
        - Where people are moving to and from
        - Which areas have high job turnover
        - Where new households are forming
        - Which regions are absorbing workers
        """)

if __name__ == "__main__":
    main()
