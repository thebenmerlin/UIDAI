# India Life-Event Insights Dashboard

> **UIDAI Data Hackathon Submission**  
> Extracting societal insights from Aadhaar activity data to reveal population-level patterns in migration, household formation, and workforce dynamics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uidai.streamlit.app)

---

## Overview

This project analyzes **anonymized Aadhaar activity data** to answer a simple question:

> *"What kind of life changes are happening across different regions of India?"*

Without tracking any individual, we reveal:
- **Where people are moving** (migration corridors)
- **Where new households are forming** (family formation patterns)
- **Where job turnover is highest** (workforce churn hotspots)
- **Which cities are absorbing workers** (urban absorption zones)

---

## Key Innovation: Life-Event Labels

We replace technical jargon with **human-readable life-event labels**:

| Technical Term | Life-Event Label |
|----------------|------------------|
| Entry Pressure / Entry-driven | New Adults Entering |
| Admin/Entry Ratio (high) | In-Migration Hub |
| Admin/Entry Ratio (low) | Out-Migration Region |
| Demographics + Biometrics | Urban Absorption Zone |
| Demo spike without enrol | Household Formation |
| Bio-Admin correlation | Workforce Churn |
| High bio activity | High Work-Related Stress |
| Balanced new adults | Emerging Urban Centre |
| Low activity | Stable Population |

A viewer can now point at any region and immediately understand: *"What life change is happening here?"*

---

## Methodology

### Three Activity Signals

| Activity | Source | What It Means |
|----------|--------|---------------|
| **New Adults Entering** | Enrolments (age 18+) | First-time adult registrations |
| **Address Changes** | Demographic updates | People moving, household changes |
| **Work Verifications** | Biometric updates | Job-related re-verification |

### Classification Logic

By comparing these signals, we classify each region:

- **More address changes than registrations** → In-Migration Hub
- **Many registrations, few address changes** → Out-Migration Region  
- **Address + work updates together** → Workforce Churn
- **Address changes without new registrations** → Household Formation
- **High work verifications** → High Work-Related Stress

---

## Data

Three datasets at **pincode-date** granularity:

| Dataset | Records | Columns |
|---------|---------|---------|
| Enrolment | 500,000 | date, state, district, pincode, age_0_5, age_5_17, age_18_greater |
| Demographic Updates | 500,000 | date, state, district, pincode, demo_age_5_17, demo_age_17_ |
| Biometric Updates | 500,000 | date, state, district, pincode, bio_age_5_17, bio_age_17_ |

**No personal data** — no names, gender, or individual identifiers.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/uidai.git
cd uidai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## Project Structure

```
uidai/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── india_states.geojson   # India map boundaries
├── Data/
│   ├── api_data_aadhar_enrolment_0_500000.csv
│   ├── api_data_aadhar_demographic_0_500000.csv
│   └── api_data_aadhar_biometric_0_500000.csv
└── README.md
```

---

## Key Findings

1. **Clear migration corridors** — Some states absorb population (high address changes), others are origin states
2. **Workforce churn hotspots** — NCR, Punjab-Haryana show synchronized job turnover signals
3. **Household formation visible** — Address changes without new enrollments reveal family formation
4. **Mature identity system** — Address changes far exceed new enrollments nationwide
5. **Regional divergence** — Northeast shows stability; South shows high work verification

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **Plotly** — Choropleth maps & charts
- **Pandas** — Data processing
- **GeoPandas** — Geographic data

---

## Team

**UIDAI Data Hackathon 2025**

---

## License

This project is submitted for the UIDAI Data Hackathon. Data is provided by UIDAI for research purposes only.
