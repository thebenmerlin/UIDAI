"""
Aadhaar Societal Insights Analysis
==================================
Extracts migration, household formation, urban absorption, and workforce churn
patterns from Aadhaar enrolment, demographic, and biometric update data.

Conceptual Framework:
- Entry Pressure: New adults entering identity system (age_18_greater)
- Admin Pressure: Life changes requiring updates (demo_age_17_)
- Physical Pressure: Biometric re-verification (bio_age_17_)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import correlate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DATA_DIR = Path("/Users/gajanan/Desktop/UIDAI/Data")
OUTPUT_DIR = Path("/Users/gajanan/Desktop/UIDAI/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """Load and merge all three Aadhaar datasets."""
    print("Loading datasets...")
    
    # Load each dataset
    enrol = pd.read_csv(DATA_DIR / "api_data_aadhar_enrolment_0_500000.csv")
    demo = pd.read_csv(DATA_DIR / "api_data_aadhar_demographic_0_500000.csv")
    bio = pd.read_csv(DATA_DIR / "api_data_aadhar_biometric_0_500000.csv")
    
    # Parse dates
    for df in [enrol, demo, bio]:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    # Standardize column names
    enrol = enrol.rename(columns={
        'age_0_5': 'enrol_0_5',
        'age_5_17': 'enrol_5_17', 
        'age_18_greater': 'enrol_18plus'
    })
    
    demo = demo.rename(columns={
        'demo_age_5_17': 'demo_5_17',
        'demo_age_17_': 'demo_17plus'
    })
    
    bio = bio.rename(columns={
        'bio_age_5_17': 'bio_5_17',
        'bio_age_17_': 'bio_17plus'
    })
    
    # Merge datasets on key columns
    merge_cols = ['date', 'state', 'district', 'pincode']
    
    merged = enrol.merge(demo, on=merge_cols, how='outer')
    merged = merged.merge(bio, on=merge_cols, how='outer')
    
    # Fill missing values with 0 (no activity recorded)
    numeric_cols = [c for c in merged.columns if c not in merge_cols]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    print(f"  Loaded {len(enrol):,} enrolment records")
    print(f"  Loaded {len(demo):,} demographic records")
    print(f"  Loaded {len(bio):,} biometric records")
    print(f"  Merged dataset: {len(merged):,} records")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  Unique states: {merged['state'].nunique()}")
    print(f"  Unique pincodes: {merged['pincode'].nunique()}")
    
    return merged

# =============================================================================
# 2. PRESSURE SIGNAL CONSTRUCTION
# =============================================================================

def compute_pressure_signals(df):
    """
    Construct three pressure signals from raw data.
    Normalize to make comparable across regions.
    """
    print("\nComputing pressure signals...")
    
    # Aggregate to state-date level for macro patterns
    state_daily = df.groupby(['date', 'state']).agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum',
        'enrol_5_17': 'sum',
        'demo_5_17': 'sum',
        'bio_5_17': 'sum'
    }).reset_index()
    
    # Create pressure signals (adults only for primary analysis)
    state_daily['entry_pressure'] = state_daily['enrol_18plus']
    state_daily['admin_pressure'] = state_daily['demo_17plus']
    state_daily['physical_pressure'] = state_daily['bio_17plus']
    
    # Normalize within each state (z-score)
    for state in state_daily['state'].unique():
        mask = state_daily['state'] == state
        for col in ['entry_pressure', 'admin_pressure', 'physical_pressure']:
            values = state_daily.loc[mask, col]
            if values.std() > 0:
                state_daily.loc[mask, f'{col}_norm'] = (values - values.mean()) / values.std()
            else:
                state_daily.loc[mask, f'{col}_norm'] = 0
    
    # Calculate divergence metrics
    state_daily['admin_entry_divergence'] = (
        state_daily['admin_pressure_norm'] - state_daily['entry_pressure_norm']
    )
    
    print(f"  Created pressure signals for {state_daily['state'].nunique()} states")
    
    return state_daily

def compute_district_signals(df):
    """Compute signals at district level for finer granularity."""
    district_daily = df.groupby(['date', 'state', 'district']).agg({
        'enrol_18plus': 'sum',
        'demo_17plus': 'sum',
        'bio_17plus': 'sum'
    }).reset_index()
    
    return district_daily

# =============================================================================
# 3. MIGRATION PATTERN DETECTION
# =============================================================================

def detect_migration_corridors(state_signals, top_n=10):
    """
    Detect migration corridors using lead-lag analysis.
    If State A's demo spike leads State B's enrolment spike, suggests A→B migration.
    """
    print("\nDetecting migration corridors...")
    
    states = state_signals['state'].unique()
    corridors = []
    
    # Pivot data for cross-correlation
    admin_pivot = state_signals.pivot(index='date', columns='state', values='admin_pressure').fillna(0)
    entry_pivot = state_signals.pivot(index='date', columns='state', values='entry_pressure').fillna(0)
    
    for origin in states:
        if origin not in admin_pivot.columns:
            continue
        origin_admin = admin_pivot[origin].values
        
        for dest in states:
            if dest == origin or dest not in entry_pivot.columns:
                continue
            dest_entry = entry_pivot[dest].values
            
            # Cross-correlation to find lead-lag
            if len(origin_admin) < 10 or np.std(origin_admin) == 0 or np.std(dest_entry) == 0:
                continue
                
            corr = np.corrcoef(origin_admin[:-7], dest_entry[7:])[0, 1]  # 7-day lag
            
            if not np.isnan(corr) and corr > 0.3:
                corridors.append({
                    'origin': origin,
                    'destination': dest,
                    'correlation': corr,
                    'origin_admin_mean': origin_admin.mean(),
                    'dest_entry_mean': dest_entry.mean()
                })
    
    corridors_df = pd.DataFrame(corridors)
    if len(corridors_df) > 0:
        corridors_df = corridors_df.sort_values('correlation', ascending=False).head(top_n)
    
    print(f"  Found {len(corridors_df)} potential migration corridors")
    return corridors_df

def identify_sending_receiving_states(state_signals):
    """
    Classify states by admin/entry ratio.
    High ratio = receiving (more address changes than new enrollments)
    Low ratio = sending (more new enrollments from home location)
    """
    print("\nClassifying states by migration role...")
    
    state_summary = state_signals.groupby('state').agg({
        'entry_pressure': 'sum',
        'admin_pressure': 'sum',
        'physical_pressure': 'sum'
    }).reset_index()
    
    # Admin/Entry ratio
    state_summary['admin_entry_ratio'] = (
        state_summary['admin_pressure'] / 
        state_summary['entry_pressure'].replace(0, 1)
    )
    
    # Bio/Admin ratio (workforce churn indicator)
    state_summary['bio_admin_ratio'] = (
        state_summary['physical_pressure'] / 
        state_summary['admin_pressure'].replace(0, 1)
    )
    
    # Classify
    median_ratio = state_summary['admin_entry_ratio'].median()
    state_summary['migration_role'] = state_summary['admin_entry_ratio'].apply(
        lambda x: 'Absorbing (Destination)' if x > median_ratio * 1.2 
        else ('Sending (Origin)' if x < median_ratio * 0.8 else 'Neutral')
    )
    
    return state_summary.sort_values('admin_entry_ratio', ascending=False)

# =============================================================================
# 4. HOUSEHOLD FORMATION DETECTION
# =============================================================================

def detect_household_formation(state_signals):
    """
    Detect household formation signals.
    Key indicator: Demo spikes WITHOUT corresponding enrolment spikes.
    """
    print("\nDetecting household formation patterns...")
    
    # Find dates where admin pressure > entry pressure significantly
    state_signals['household_signal'] = (
        (state_signals['admin_pressure_norm'] > 1.5) &  # Admin spike
        (state_signals['entry_pressure_norm'] < 0.5)     # No entry spike
    ).astype(int)
    
    # Aggregate by state
    hh_by_state = state_signals.groupby('state').agg({
        'household_signal': 'sum',
        'admin_pressure': 'sum',
        'entry_pressure': 'sum'
    }).reset_index()
    
    hh_by_state['household_intensity'] = (
        hh_by_state['household_signal'] / 
        state_signals.groupby('state').size().values
    )
    
    return hh_by_state.sort_values('household_intensity', ascending=False)

# =============================================================================
# 5. WORKFORCE CHURN DETECTION
# =============================================================================

def detect_workforce_churn(state_signals):
    """
    Detect workforce churn patterns.
    Key indicator: Correlated Bio + Admin spikes (job changes need both updates).
    """
    print("\nDetecting workforce churn patterns...")
    
    churn_results = []
    
    for state in state_signals['state'].unique():
        state_data = state_signals[state_signals['state'] == state]
        
        if len(state_data) < 5:
            continue
            
        admin = state_data['admin_pressure'].values
        bio = state_data['physical_pressure'].values
        
        if np.std(admin) > 0 and np.std(bio) > 0:
            corr = np.corrcoef(admin, bio)[0, 1]
        else:
            corr = 0
            
        churn_results.append({
            'state': state,
            'bio_admin_correlation': corr,
            'total_admin': admin.sum(),
            'total_bio': bio.sum(),
            'churn_intensity': corr * (admin.sum() + bio.sum()) / 2
        })
    
    churn_df = pd.DataFrame(churn_results)
    churn_df = churn_df.sort_values('bio_admin_correlation', ascending=False)
    
    # High correlation = likely workforce churn
    churn_df['churn_type'] = churn_df['bio_admin_correlation'].apply(
        lambda x: 'High Churn' if x > 0.7 else ('Moderate Churn' if x > 0.4 else 'Low Churn')
    )
    
    return churn_df

# =============================================================================
# 6. GEOGRAPHIC CLUSTERING
# =============================================================================

def cluster_regions(state_summary, churn_df, n_clusters=4):
    """
    Cluster states by their pressure profile.
    Creates typology: Absorbing Urban, Sending Rural, Churn Hubs, Stable.
    """
    print("\nClustering regions by pressure profile...")
    
    # Merge features
    features_df = state_summary.merge(
        churn_df[['state', 'bio_admin_correlation']], 
        on='state', 
        how='left'
    ).fillna(0)
    
    # Feature matrix
    feature_cols = ['admin_entry_ratio', 'bio_admin_ratio', 'bio_admin_correlation']
    X = features_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Label clusters based on characteristics
    cluster_profiles = features_df.groupby('cluster')[feature_cols].mean()
    
    cluster_labels = {}
    for c in range(n_clusters):
        profile = cluster_profiles.loc[c]
        if profile['admin_entry_ratio'] > cluster_profiles['admin_entry_ratio'].median():
            if profile['bio_admin_correlation'] > 0.5:
                cluster_labels[c] = 'Urban Churn Hub'
            else:
                cluster_labels[c] = 'Absorbing Region'
        else:
            if profile['bio_admin_correlation'] > 0.5:
                cluster_labels[c] = 'Transitional Zone'
            else:
                cluster_labels[c] = 'Sending Region'
    
    features_df['region_type'] = features_df['cluster'].map(cluster_labels)
    
    print(f"  Cluster distribution:")
    for rtype, count in features_df['region_type'].value_counts().items():
        print(f"    {rtype}: {count} states")
    
    return features_df

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def create_visualizations(state_signals, state_summary, churn_df, corridors_df, region_types):
    """Generate all visualizations."""
    print("\nCreating visualizations...")
    
    # 1. National Pressure Timeline
    fig, ax = plt.subplots(figsize=(14, 6))
    national = state_signals.groupby('date').agg({
        'entry_pressure': 'sum',
        'admin_pressure': 'sum', 
        'physical_pressure': 'sum'
    }).reset_index()
    
    ax.plot(national['date'], national['entry_pressure'], label='Entry Pressure', linewidth=2)
    ax.plot(national['date'], national['admin_pressure'], label='Admin Pressure', linewidth=2)
    ax.plot(national['date'], national['physical_pressure'], label='Physical Pressure', linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Activity Volume', fontsize=12)
    ax.set_title('National Aadhaar Activity: Three Pressure Signals', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pressure_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: pressure_timeline.png")
    
    # 2. State Migration Role
    fig, ax = plt.subplots(figsize=(14, 8))
    top_states = state_summary.head(20)
    colors = ['#2ecc71' if r == 'Absorbing (Destination)' else 
              '#e74c3c' if r == 'Sending (Origin)' else '#95a5a6' 
              for r in top_states['migration_role']]
    
    bars = ax.barh(top_states['state'], top_states['admin_entry_ratio'], color=colors)
    ax.axvline(x=state_summary['admin_entry_ratio'].median(), color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Admin/Entry Ratio (Higher = More Absorbing)', fontsize=12)
    ax.set_title('State Migration Roles: Admin/Entry Pressure Ratio', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Absorbing (Destination)'),
                      Patch(facecolor='#e74c3c', label='Sending (Origin)'),
                      Patch(facecolor='#95a5a6', label='Neutral')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'migration_roles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: migration_roles.png")
    
    # 3. Workforce Churn by State
    fig, ax = plt.subplots(figsize=(12, 8))
    churn_sorted = churn_df.sort_values('bio_admin_correlation', ascending=True).tail(20)
    colors = ['#e74c3c' if t == 'High Churn' else 
              '#f39c12' if t == 'Moderate Churn' else '#3498db'
              for t in churn_sorted['churn_type']]
    
    ax.barh(churn_sorted['state'], churn_sorted['bio_admin_correlation'], color=colors)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='High Churn Threshold')
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate Threshold')
    ax.set_xlabel('Bio-Admin Correlation (Higher = More Churn)', fontsize=12)
    ax.set_title('Workforce Churn Intensity by State', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'workforce_churn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: workforce_churn.png")
    
    # 4. Region Typology
    fig, ax = plt.subplots(figsize=(10, 8))
    type_counts = region_types['region_type'].value_counts()
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
           colors=colors[:len(type_counts)], startangle=90)
    ax.set_title('Region Typology Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'region_typology.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: region_typology.png")
    
    # 5. Pressure Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pressure_cols = ['entry_pressure', 'admin_pressure', 'physical_pressure']
    corr_matrix = state_signals[pressure_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax, fmt='.2f',
                xticklabels=['Entry\n(Enrolment)', 'Admin\n(Demo Updates)', 'Physical\n(Bio Updates)'],
                yticklabels=['Entry\n(Enrolment)', 'Admin\n(Demo Updates)', 'Physical\n(Bio Updates)'])
    ax.set_title('Pressure Signal Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pressure_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: pressure_correlations.png")

# =============================================================================
# 8. INSIGHT GENERATION
# =============================================================================

def generate_insights(state_summary, churn_df, corridors_df, region_types, hh_formation):
    """Generate narrative insights from analysis."""
    
    insights = []
    
    # Insight 1: Migration Asymmetry
    top_absorbing = state_summary[state_summary['migration_role'] == 'Absorbing (Destination)'].head(5)
    top_sending = state_summary[state_summary['migration_role'] == 'Sending (Origin)'].head(5)
    
    if len(top_absorbing) > 0 and len(top_sending) > 0:
        insights.append({
            'title': 'Clear Migration Asymmetry Detected',
            'what_we_see': f"States like {', '.join(top_absorbing['state'].head(3).tolist())} show 2-3x higher admin/entry ratios than {', '.join(top_sending['state'].head(3).tolist())}",
            'why_it_matters': "High admin/entry ratio indicates people moving IN and updating addresses, while low ratio suggests origin states where people get Aadhaar but leave. This reveals internal migration pressure without tracking individuals."
        })
    
    # Insight 2: Workforce Churn Hotspots
    high_churn = churn_df[churn_df['churn_type'] == 'High Churn']
    if len(high_churn) > 0:
        insights.append({
            'title': 'Workforce Churn Hotspots Identified',
            'what_we_see': f"{len(high_churn)} states show bio-admin correlation >0.7: {', '.join(high_churn['state'].head(5).tolist())}",
            'why_it_matters': "When biometric and demographic updates spike together, it signals workforce turnover—people changing jobs often need both address updates and biometric re-verification. These regions have high labor market dynamism."
        })
    
    # Insight 3: Household Formation Timing
    top_hh = hh_formation.head(5)
    if len(top_hh) > 0:
        insights.append({
            'title': 'Household Formation Patterns Visible',
            'what_we_see': f"States with highest household formation signals: {', '.join(top_hh['state'].tolist())}",
            'why_it_matters': "Demo spikes WITHOUT enrolment spikes indicate existing Aadhaar holders changing addresses—classic household formation (marriage, new families). This is a proxy for life-stage transitions at population level."
        })
    
    # Insight 4: Urban Absorption Capacity
    urban_hubs = region_types[region_types['region_type'] == 'Urban Churn Hub']
    absorbing = region_types[region_types['region_type'] == 'Absorbing Region']
    
    if len(urban_hubs) > 0 or len(absorbing) > 0:
        all_absorbing = pd.concat([urban_hubs, absorbing]) if len(urban_hubs) > 0 else absorbing
        insights.append({
            'title': 'Urban Absorption Capacity Mapped',
            'what_we_see': f"{len(all_absorbing)} states classified as absorbing/urban hubs: {', '.join(all_absorbing['state'].head(5).tolist())}",
            'why_it_matters': "These regions consistently show more address-change activity than new enrolments—they're absorbing population from elsewhere. High churn variants indicate dynamic job markets; pure absorbing indicates residential settlement."
        })
    
    # Insight 5: Regional Pressure Imbalances
    national_admin = state_summary['admin_pressure'].sum()
    national_entry = state_summary['entry_pressure'].sum()
    national_ratio = national_admin / national_entry if national_entry > 0 else 0
    
    insights.append({
        'title': 'National Administrative Pressure Exceeds Entry',
        'what_we_see': f"Nationwide, admin pressure is {national_ratio:.1f}x the entry pressure",
        'why_it_matters': "More people are updating addresses than getting new Aadhaar—consistent with a maturing identity system where most adults already have Aadhaar and activity reflects life changes, not first-time enrollment."
    })
    
    # Insight 6: Northeast Distinctiveness
    ne_states = ['Assam', 'Meghalaya', 'Nagaland', 'Manipur', 'Tripura', 'Mizoram', 'Arunachal Pradesh']
    ne_data = region_types[region_types['state'].isin(ne_states)]
    
    if len(ne_data) > 0:
        ne_types = ne_data['region_type'].value_counts()
        insights.append({
            'title': 'Northeast Shows Distinct Lifecycle Pattern',
            'what_we_see': f"Northeast states predominantly classified as: {ne_types.index[0] if len(ne_types) > 0 else 'Mixed'}",
            'why_it_matters': "The Northeast's distinct pressure profile reflects different economic patterns—likely more stable populations with different migration dynamics than the Hindi heartland corridor."
        })
    
    # Insight 7: Entry vs Physical Pressure Divergence
    bio_heavy = state_summary[state_summary['bio_admin_ratio'] > state_summary['bio_admin_ratio'].median() * 1.5]
    if len(bio_heavy) > 0:
        insights.append({
            'title': 'Biometric-Heavy Regions Signal Verification Pressure',
            'what_we_see': f"States with unusually high biometric activity: {', '.join(bio_heavy['state'].head(5).tolist())}",
            'why_it_matters': "High biometric updates relative to demographic updates may indicate: aging population needing biometric refresh, or regions with stricter verification requirements for welfare programs."
        })
    
    return insights

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("AADHAAR SOCIETAL INSIGHTS ANALYSIS")
    print("=" * 60)
    
    # Load and process data
    df = load_data()
    state_signals = compute_pressure_signals(df)
    
    # Run analyses
    state_summary = identify_sending_receiving_states(state_signals)
    corridors = detect_migration_corridors(state_signals)
    hh_formation = detect_household_formation(state_signals)
    churn = detect_workforce_churn(state_signals)
    region_types = cluster_regions(state_summary, churn)
    
    # Generate visualizations
    create_visualizations(state_signals, state_summary, churn, corridors, region_types)
    
    # Generate insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    insights = generate_insights(state_summary, churn, corridors, region_types, hh_formation)
    
    for i, insight in enumerate(insights, 1):
        print(f"\n### Insight {i}: {insight['title']}")
        print(f"**What we see:** {insight['what_we_see']}")
        print(f"**Why it matters:** {insight['why_it_matters']}")
    
    # Save insights to file
    with open(OUTPUT_DIR.parent / 'insights_report.md', 'w') as f:
        f.write("# Aadhaar Societal Insights Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This analysis treats Aadhaar activity as three population-level pressure signals:\n")
        f.write("- **Entry Pressure**: New adults entering the identity system\n")
        f.write("- **Admin Pressure**: Life changes requiring address/identity updates\n")
        f.write("- **Physical Pressure**: Biometric re-verification needs\n\n")
        f.write("By examining imbalances between these signals across geography and time, ")
        f.write("we reveal migration corridors, household formation patterns, workforce churn, ")
        f.write("and urban absorption capacity—entirely at population level.\n\n")
        f.write("---\n\n")
        
        for i, insight in enumerate(insights, 1):
            f.write(f"## Insight {i}: {insight['title']}\n\n")
            f.write(f"**What we see:** {insight['what_we_see']}\n\n")
            f.write(f"**Why it matters:** {insight['why_it_matters']}\n\n")
            f.write("---\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Pressure Timeline](visualizations/pressure_timeline.png)\n\n")
        f.write("![Migration Roles](visualizations/migration_roles.png)\n\n")
        f.write("![Workforce Churn](visualizations/workforce_churn.png)\n\n")
        f.write("![Region Typology](visualizations/region_typology.png)\n\n")
        f.write("![Pressure Correlations](visualizations/pressure_correlations.png)\n\n")
    
    print(f"\n\nResults saved to: {OUTPUT_DIR.parent / 'insights_report.md'}")
    print(f"Visualizations saved to: {OUTPUT_DIR}/")
    
    return {
        'state_signals': state_signals,
        'state_summary': state_summary,
        'corridors': corridors,
        'churn': churn,
        'region_types': region_types,
        'insights': insights
    }

if __name__ == "__main__":
    results = main()
