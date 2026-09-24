# =============================================================================
# NYAMBENE SUBCOUNTY HOSPITAL
# MATERNAL HEALTH ANALYTICS AND DECISION SUPPORT SYSTEM
# COLOR-THEMED STREAMLIT INTERFACE VERSION
# =============================================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nyambene Maternal Health DSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_data():

    path = "data/nyambene_final_dashboard_dataset.csv"

    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        st.stop()

    return pd.read_csv(path)


df = load_data()


# =============================================================================
# LOAD MODELS
# =============================================================================

MODEL_DIR = "models"


@st.cache_resource
def load_models():

    logistic_path = os.path.join(
        MODEL_DIR,
        "delivery_logistic_model.pkl"
    )

    rf_path = os.path.join(
        MODEL_DIR,
        "delivery_random_forest_model.pkl"
    )

    hgb_path = os.path.join(
        MODEL_DIR,
        "delivery_hgb_model.pkl"
    )

    missing_models = [
        path
        for path in [
            logistic_path,
            rf_path,
            hgb_path
        ]
        if not os.path.exists(path)
    ]

    if missing_models:

        st.error("One or more model files are missing:")

        for path in missing_models:
            st.write(f"- {path}")

        st.stop()

    logistic = joblib.load(logistic_path)
    random_forest = joblib.load(rf_path)
    hgb = joblib.load(hgb_path)

    return logistic, random_forest, hgb


delivery_logistic, delivery_rf, delivery_hgb = load_models()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def percentage(numerator, denominator):

    if denominator == 0:
        return 0

    return numerator / denominator * 100


def chart_layout(fig, height=430):

    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        ),
        legend_title_text="",
    )

    return fig


def safe_mean(series):

    if series.dropna().empty:
        return "—"

    return f"{series.mean():.2f}"


def page_header(title, description):

    st.title(title)
    st.caption(description)

    st.divider()


# =============================================================================
# PAGE COLOR THEMES
# =============================================================================

PAGE_THEMES = {
    "🏠 Hospital Overview": {"main": "#2563EB", "soft": "#EFF6FF", "accent": "#1D4ED8"},
    "👩 Maternal & ANC": {"main": "#7C3AED", "soft": "#F5F3FF", "accent": "#6D28D9"},
    "🤰 Pregnancy & Newborn": {"main": "#0F766E", "soft": "#F0FDFA", "accent": "#115E59"},
    "🚼 Delivery Analysis": {"main": "#EA580C", "soft": "#FFF7ED", "accent": "#C2410C"},
    "🩸 Maternal Safety": {"main": "#DC2626", "soft": "#FEF2F2", "accent": "#B91C1C"},
    "🔎 Data Quality": {"main": "#D97706", "soft": "#FFFBEB", "accent": "#B45309"},
    "📊 Statistical Evidence": {"main": "#4F46E5", "soft": "#EEF2FF", "accent": "#3730A3"},
    "🤖 AI — Delivery Prediction": {"main": "#1D4ED8", "soft": "#EFF6FF", "accent": "#1E3A8A"},
    "👶 AI — Baby Risk Prediction": {"main": "#059669", "soft": "#ECFDF5", "accent": "#047857"},
}

FLOATING_OBJECTS = {
    "🏠 Hospital Overview": ("🏥", "Hospital", "Health Analytics"),
    "👩 Maternal & ANC": ("👩‍🍼", "Maternal Care", "ANC Monitoring"),
    "🤰 Pregnancy & Newborn": ("🤰", "Pregnancy", "Newborn Health"),
    "🚼 Delivery Analysis": ("👶", "Delivery", "Outcome Analysis"),
    "🩸 Maternal Safety": ("❤️", "Maternal Safety", "Clinical Monitoring"),
    "🔎 Data Quality": ("🔎", "Data Quality", "Record Verification"),
    "📊 Statistical Evidence": ("📊", "Statistics", "Evidence & Patterns"),
    "🤖 AI — Delivery Prediction": ("🤖", "AI Prediction", "Delivery Support"),
    "👶 AI — Baby Risk Prediction": ("🧠", "AI Screening", "Baby Risk"),
}


def apply_page_theme(page_name):
    theme = PAGE_THEMES.get(page_name, PAGE_THEMES["🏠 Hospital Overview"])
    main = theme["main"]
    soft = theme["soft"]
    accent = theme["accent"]
    st.markdown(f"""
    <style>
        :root {{ --page-main: {main}; --page-soft: {soft}; --page-accent: {accent}; }}
        .stApp {{ background: linear-gradient(180deg, {soft} 0%, #FFFFFF 30%); }}
        [data-testid="stSidebar"] {{ border-right: 5px solid {main}; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {soft}; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {accent}; }}
        .stButton > button {{ border-color: {main}; color: {accent}; }}
        .stButton > button:hover {{ border-color: {main}; color: white; background: {main}; }}
        div[data-testid="stMetric"] {{ background: white; border-top: 5px solid {main}; border-radius: 12px; padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
        div[data-testid="stMetricValue"] {{ color: {accent}; }}
        .stProgress > div > div > div > div {{ background-color: {main}; }}
        .stTabs [aria-selected="true"] {{ color: {accent}; border-bottom-color: {main}; }}
        div[data-testid="stDataFrame"] {{ border-top: 3px solid {main}; }}
        h1, h2, h3 {{ color: {accent}; }}
        .nyambene-floating-object {{
            position: fixed; top: 92px; right: 24px; width: 108px; height: 108px;
            border-radius: 24px; background: rgba(255,255,255,.94);
            border: 2px solid {main}; box-shadow: 0 12px 32px rgba(15,23,42,.16);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            z-index: 999; pointer-events: none; animation: nyambeneFloat 4s ease-in-out infinite;
        }}
        .nyambene-floating-icon {{ font-size: 38px; line-height: 1; margin-bottom: 7px; }}
        .nyambene-floating-title {{ color: {accent}; font-size: 12px; font-weight: 800; text-align: center; line-height: 1.15; }}
        .nyambene-floating-subtitle {{ color: {accent}; opacity: .68; font-size: 9px; text-align: center; margin-top: 3px; line-height: 1.1; }}
        @keyframes nyambeneFloat {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-9px); }} }}
        @media (max-width: 1100px) {{
            .nyambene-floating-object {{ width: 88px; height: 88px; right: 14px; top: 88px; border-radius: 20px; }}
            .nyambene-floating-icon {{ font-size: 30px; }} .nyambene-floating-title {{ font-size: 10px; }} .nyambene-floating-subtitle {{ display:none; }}
        }}
        @media (max-width: 768px) {{ .nyambene-floating-object {{ display:none; }} }}
        .theme-banner {{ background: {main}; color: white; padding: 14px 20px; border-radius: 14px; margin: 4px 0 18px 0; font-weight: 700; box-shadow: 0 4px 14px rgba(0,0,0,.10); }}
        .theme-banner small {{ opacity: .9; font-weight: 400; }}
    </style>
    <div class="theme-banner">Nyambene Subcounty Hospital <small>• {page_name}</small></div>
    """, unsafe_allow_html=True)

    icon, title, subtitle = FLOATING_OBJECTS.get(page_name, ("🏥", "Nyambene DSS", "Health Analytics"))
    st.markdown(
        f"""<div class="nyambene-floating-object">
            <div class="nyambene-floating-icon">{icon}</div>
            <div class="nyambene-floating-title">{title}</div>
            <div class="nyambene-floating-subtitle">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🏥 Nyambene Hospital")

    st.caption(
        "Maternal Health Analytics & Decision Support System"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select page",
        [
            "🏠 Hospital Overview",
            "👩 Maternal & ANC",
            "🤰 Pregnancy & Newborn",
            "🚼 Delivery Analysis",
            "🩸 Maternal Safety",
            "🔎 Data Quality",
            "📊 Statistical Evidence",
            "🤖 AI — Delivery Prediction",
            "👶 AI — Baby Risk Prediction",
        ],
    )

    st.divider()

    # -------------------------------------------------------------------------
    # FILTERS
    # -------------------------------------------------------------------------

    if page not in [
        "📊 Statistical Evidence",
        "🤖 AI — Delivery Prediction",
        "👶 AI — Baby Risk Prediction",
    ]:

        st.subheader("🔍 Filters")

        age_options = sorted(
            df["Age_Group"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        delivery_options = sorted(
            df["Delivery_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_age_groups = st.multiselect(
            "Maternal Age Group",
            options=age_options,
            default=age_options,
        )

        selected_delivery_types = st.multiselect(
            "Delivery Type",
            options=delivery_options,
            default=delivery_options,
        )

        filtered_df = df[
            df["Age_Group"]
            .astype(str)
            .isin(selected_age_groups)
            &
            (
                df["Delivery_Type"]
                .astype(str)
                .isin(selected_delivery_types)
                |
                df["Delivery_Type"].isna()
            )
        ].copy()

    else:

        filtered_df = df.copy()

    st.divider()

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} records"
    )


apply_page_theme(page)

# =============================================================================
# MAIN TITLE
# =============================================================================

st.title("🏥 Nyambene Subcounty Hospital")

st.subheader(
    "Maternal Health Analytics & Decision Support System"
)

st.caption(
    "Facility-level maternal health analytics, statistical evidence and "
    "exploratory machine-learning decision support."
)

st.divider()


# =============================================================================
# PAGE 1 — HOSPITAL OVERVIEW
# =============================================================================

if page == "🏠 Hospital Overview":

    page_header(
        "🏠 Hospital Overview",
        "High-level summary of maternal, pregnancy, delivery, newborn and data-quality indicators.",
    )

    # -------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------

    total_records = len(filtered_df)

    adolescents = (
        filtered_df["AGE"] < 18
    ).sum()

    normal = (
        filtered_df["Delivery_Type"] == "Normal"
    ).sum()

    caesarean = (
        filtered_df["Delivery_Type"] == "Caesarean"
    ).sum()

    lbw = (
        filtered_df["Low_Birth_Weight"] == "Yes"
    ).sum()

    deformity = (
        filtered_df["Deformity"] == "Yes"
    ).sum()

    check_records = (
        filtered_df["Data_Quality_Status"] == "CHECK"
    ).sum()

    valid_bw = (
        filtered_df["Birth_weight_in_grams"]
        .notna()
        .sum()
    )

    valid_delivery = (
        filtered_df["Delivery_Type"]
        .notna()
        .sum()
    )

    lbw_rate = percentage(
        lbw,
        valid_bw
    )

    cs_rate = percentage(
        caesarean,
        valid_delivery
    )

    # -------------------------------------------------------------------------
    # KPIs
    # -------------------------------------------------------------------------

    st.subheader("📌 Core Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Maternal Records",
        f"{total_records:,}"
    )

    c2.metric(
        "Normal Deliveries",
        f"{normal:,}"
    )

    c3.metric(
        "Caesarean Deliveries",
        f"{caesarean:,}"
    )

    c4.metric(
        "Low Birth Weight",
        f"{lbw:,}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "LBW Rate",
        f"{lbw_rate:.2f}%"
    )

    c2.metric(
        "Caesarean Share",
        f"{cs_rate:.2f}%"
    )

    c3.metric(
        "Adolescent Mothers",
        f"{adolescents:,}"
    )

    c4.metric(
        "Records to Verify",
        f"{check_records:,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # CHARTS
    # -------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        delivery_counts = (
            filtered_df[
                "Delivery_Type"
            ]
            .value_counts()
            .rename_axis("Delivery_Type")
            .reset_index(name="Count")
        )

        fig = px.pie(
            delivery_counts,
            names="Delivery_Type",
            values="Count",
            hole=0.45,
            title="Delivery Mode Distribution",
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

    with col2:

        age_counts = (
            filtered_df[
                "Age_Group"
            ]
            .value_counts()
            .rename_axis("Age_Group")
            .reset_index(name="Count")
        )

        fig = px.bar(
            age_counts,
            x="Age_Group",
            y="Count",
            text="Count",
            title="Maternal Age Groups",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------------

    st.subheader("📋 Dataset Summary")

    summary_table = pd.DataFrame(
        {
            "Indicator": [
                "Total records",
                "Adolescent mothers",
                "Normal deliveries",
                "Caesarean deliveries",
                "Low birth weight",
                "Birth deformity",
                "Records requiring verification",
            ],
            "Count": [
                total_records,
                adolescents,
                normal,
                caesarean,
                lbw,
                deformity,
                check_records,
            ],
        }
    )

    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These indicators describe the observed records. They do not establish "
        "causal relationships."
    )


# =============================================================================
# PAGE 2 — MATERNAL & ANC
# =============================================================================

elif page == "👩 Maternal & ANC":

    page_header(
        "👩 Maternal & ANC",
        "Maternal age profile and antenatal-care utilization patterns.",
    )

    valid_age = filtered_df[
        "AGE"
    ].dropna()

    valid_anc = filtered_df[
        "ANC"
    ].dropna()

    adolescents = (
        filtered_df["AGE"] < 18
    ).sum()

    age_35_plus = (
        filtered_df["AGE"] >= 35
    ).sum()

    st.subheader("📌 Maternal Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Age-valid Records",
        f"{len(valid_age):,}"
    )

    c2.metric(
        "Mean Age",
        safe_mean(valid_age)
    )

    c3.metric(
        "Adolescent Mothers",
        f"{adolescents:,}"
    )

    c4.metric(
        "Mothers 35+",
        f"{age_35_plus:,}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ANC-valid Records",
        f"{len(valid_anc):,}"
    )

    c2.metric(
        "Median ANC",
        f"{valid_anc.median():.0f}"
        if len(valid_anc)
        else "—"
    )

    c3.metric(
        "Missing ANC",
        f"{filtered_df['ANC'].isna().sum():,}"
    )

    c4.metric(
        "ANC = 0",
        f"{(filtered_df['ANC'] == 0).sum():,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # AGE + ANC CHARTS
    # -------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered_df,
            x="AGE",
            nbins=20,
            title="Maternal Age Distribution",
        )

        fig.update_xaxes(
            title="Age (years)"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with col2:

        anc_counts = (
            filtered_df[
                "ANC_Group"
            ]
            .value_counts()
            .rename_axis("ANC_Group")
            .reset_index(name="Count")
        )

        fig = px.bar(
            anc_counts,
            x="ANC_Group",
            y="Count",
            text="Count",
            title="ANC Visit Groups",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # ANC VS DELIVERY
    # -------------------------------------------------------------------------

    st.subheader("📊 ANC Group vs Delivery Mode")

    anc_delivery = pd.crosstab(
        filtered_df["ANC_Group"],
        filtered_df["Delivery_Type"],
    ).reset_index()

    delivery_columns = [
        c
        for c in anc_delivery.columns
        if c != "ANC_Group"
    ]

    if delivery_columns:

        fig = px.bar(
            anc_delivery,
            x="ANC_Group",
            y=delivery_columns,
            barmode="group",
            title="ANC Group by Delivery Type",
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # AGE SUMMARY
    # -------------------------------------------------------------------------

    st.subheader("👩 Maternal Age Group Summary")

    age_summary = (
        filtered_df
        .groupby(
            "Age_Group",
            dropna=False
        )
        .agg(
            Records=("AGE", "size"),
            Mean_Age=("AGE", "mean"),
            LBW_Cases=(
                "Low_Birth_Weight",
                lambda x: (x == "Yes").sum()
            ),
        )
        .reset_index()
    )

    age_summary["Mean_Age"] = (
        age_summary["Mean_Age"]
        .round(2)
    )

    st.dataframe(
        age_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "ANC has notable missingness. ANC-based observations should therefore "
        "be interpreted using records with available ANC information."
    )


# =============================================================================
# PAGE 3 — PREGNANCY & NEWBORN
# =============================================================================

elif page == "🤰 Pregnancy & Newborn":

    page_header(
        "🤰 Pregnancy & Newborn",
        "Gestational age, birth weight and low-birth-weight indicators.",
    )

    valid_gestation = filtered_df[
        "Gestation"
    ].notna().sum()

    valid_bw = filtered_df[
        "Birth_weight_in_grams"
    ].notna().sum()

    lbw = (
        filtered_df[
            "Low_Birth_Weight"
        ] == "Yes"
    ).sum()

    deformity = (
        filtered_df[
            "Deformity"
        ] == "Yes"
    ).sum()

    st.subheader("📌 Pregnancy & Newborn Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Valid Gestation",
        f"{valid_gestation:,}"
    )

    c2.metric(
        "Valid Birth Weight",
        f"{valid_bw:,}"
    )

    c3.metric(
        "Low Birth Weight",
        f"{lbw:,}"
    )

    c4.metric(
        "Birth Deformity",
        f"{deformity:,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # GESTATION + BIRTH WEIGHT
    # -------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gestation_counts = (
            filtered_df[
                "Gestation_Category"
            ]
            .value_counts()
            .rename_axis("Gestation_Category")
            .reset_index(name="Count")
        )

        fig = px.bar(
            gestation_counts,
            x="Gestation_Category",
            y="Count",
            text="Count",
            title="Gestation Category Distribution",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            filtered_df,
            x="Birth_weight_in_grams",
            nbins=30,
            title="Birth Weight Distribution",
        )

        fig.update_xaxes(
            title="Birth weight (g)"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # LBW BY GESTATION
    # -------------------------------------------------------------------------

    st.subheader("⚖️ Low Birth Weight by Gestation")

    lbw_gestation = (
        filtered_df
        .groupby(
            "Gestation_Category"
        )
        .agg(
            Total=(
                "Low_Birth_Weight",
                "count"
            ),
            LBW=(
                "Low_Birth_Weight",
                lambda x: (x == "Yes").sum()
            ),
        )
        .reset_index()
    )

    lbw_gestation["LBW_Rate"] = (
        lbw_gestation["LBW"]
        /
        lbw_gestation["Total"]
        *
        100
    ).round(2)

    fig = px.bar(
        lbw_gestation,
        x="Gestation_Category",
        y="LBW_Rate",
        text="LBW_Rate",
        title="Observed LBW Rate by Gestation",
    )

    fig.update_yaxes(
        title="LBW rate (%)"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )

    st.dataframe(
        lbw_gestation,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Gestational-age category was statistically associated with low birth "
        "weight in the full analysis. This is an observed association, not proof "
        "of causation."
    )


# =============================================================================
# PAGE 4 — DELIVERY ANALYSIS
# =============================================================================

elif page == "🚼 Delivery Analysis":

    page_header(
        "🚼 Delivery Analysis",
        "Observed delivery-mode patterns and relationships with recorded maternal and pregnancy variables.",
    )

    valid_delivery = filtered_df[
        "Delivery_Type"
    ].notna().sum()

    normal = (
        filtered_df[
            "Delivery_Type"
        ] == "Normal"
    ).sum()

    caesarean = (
        filtered_df[
            "Delivery_Type"
        ] == "Caesarean"
    ).sum()

    breech = (
        filtered_df[
            "Delivery_Type"
        ] == "Breech"
    ).sum()

    assisted = (
        filtered_df[
            "Delivery_Type"
        ] == "Assisted_Vaginal"
    ).sum()

    st.subheader("📌 Delivery Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recorded Deliveries",
        f"{valid_delivery:,}"
    )

    c2.metric(
        "Normal",
        f"{normal:,}"
    )

    c3.metric(
        "Caesarean",
        f"{caesarean:,}"
    )

    c4.metric(
        "Breech / Assisted",
        f"{breech + assisted:,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # DELIVERY DISTRIBUTION
    # -------------------------------------------------------------------------

    delivery_counts = (
        filtered_df[
            "Delivery_Type"
        ]
        .value_counts()
        .rename_axis("Delivery_Type")
        .reset_index(name="Count")
    )

    fig = px.bar(
        delivery_counts,
        x="Delivery_Type",
        y="Count",
        text="Count",
        title="Delivery Mode Distribution",
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )

    # -------------------------------------------------------------------------
    # DIAGNOSIS VS DELIVERY
    # -------------------------------------------------------------------------

    st.subheader("🩺 Diagnosis vs Delivery Mode")

    diagnosis_delivery = pd.crosstab(
        filtered_df["Diagnosis"],
        filtered_df["Delivery_Type"],
    ).reset_index()

    delivery_columns = [
        c
        for c in diagnosis_delivery.columns
        if c != "Diagnosis"
    ]

    if delivery_columns:

        fig = px.bar(
            diagnosis_delivery,
            x="Diagnosis",
            y=delivery_columns,
            barmode="stack",
            title="Diagnosis Distribution by Delivery Mode",
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # GESTATION VS DELIVERY
    # -------------------------------------------------------------------------

    st.subheader("🤰 Gestation vs Delivery")

    gestation_delivery = pd.crosstab(
        filtered_df["Gestation_Category"],
        filtered_df["Delivery_Type"],
    ).reset_index()

    delivery_columns = [
        c
        for c in gestation_delivery.columns
        if c != "Gestation_Category"
    ]

    if delivery_columns:

        fig = px.bar(
            gestation_delivery,
            x="Gestation_Category",
            y=delivery_columns,
            barmode="group",
            title="Gestation Category by Delivery Mode",
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.info(
        "In the full statistical analysis, diagnosis, ANC group and gestation "
        "category showed statistically significant associations with delivery mode. "
        "Association does not establish causation."
    )


# =============================================================================
# PAGE 5 — MATERNAL SAFETY
# =============================================================================

elif page == "🩸 Maternal Safety":

    page_header(
        "🩸 Maternal Safety Monitoring",
        "Exploratory monitoring of recorded blood loss and labour-duration indicators.",
    )

    blood_500 = (
        filtered_df[
            "Blood Loss"
        ] >= 500
    ).sum()

    blood_2000 = (
        filtered_df[
            "Blood Loss"
        ] > 2000
    ).sum()

    labour_24 = (
        filtered_df[
            "Labour min"
        ] > 1440
    ).sum()

    valid_blood = (
        filtered_df[
            "Blood Loss"
        ].notna()
        .sum()
    )

    valid_labour = (
        filtered_df[
            "Labour min"
        ].notna()
        .sum()
    )

    st.subheader("📌 Monitoring Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Valid Blood Loss",
        f"{valid_blood:,}"
    )

    c2.metric(
        "Blood Loss ≥500 ml",
        f"{blood_500:,}"
    )

    c3.metric(
        "Blood Loss >2000 ml",
        f"{blood_2000:,}"
    )

    c4.metric(
        "Labour >24 Hours",
        f"{labour_24:,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # CHARTS
    # -------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered_df,
            x="Blood Loss",
            nbins=30,
            title="Blood Loss Distribution",
        )

        fig.update_xaxes(
            title="Blood loss (ml)"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with col2:

        labour_df = filtered_df[
            filtered_df["Labour min"].notna()
        ].copy()

        fig = px.histogram(
            labour_df,
            x="Labour min",
            nbins=30,
            title="Labour Duration Distribution",
        )

        fig.update_xaxes(
            title="Labour duration (minutes)"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.warning(
        "Blood loss ≥500 ml is used here as an exploratory monitoring indicator. "
        "It is not automatically classified as a clinical diagnosis. Labour >24 "
        "hours is similarly a duration flag, not by itself a diagnosis."
    )

    # -------------------------------------------------------------------------
    # MONITORING FLAGS
    # -------------------------------------------------------------------------

    st.subheader("🔎 Records Requiring Verification")

    verification_df = filtered_df[
        filtered_df[
            "Monitoring_Flag"
        ].fillna("OK") != "OK"
    ].copy()

    if verification_df.empty:

        st.success(
            "No monitoring flags are present under the current filters."
        )

    else:

        display_columns = [
            "Record_ID",
            "AGE",
            "Blood Loss",
            "Labour min",
            "Birth_weight_in_grams",
            "Data_Quality_Status",
            "Monitoring_Flag",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in verification_df.columns
        ]

        st.dataframe(
            verification_df[
                available_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# PAGE 6 — DATA QUALITY
# =============================================================================

elif page == "🔎 Data Quality":

    page_header(
        "🔎 Data Quality",
        "Completeness, unusual values, verification flags and duplicate-record review.",
    )

    quality_columns = [
        "AGE",
        "ANC",
        "Parity",
        "Gravidae",
        "Diagnosis",
        "Gestation",
        "Mode_Of_Delivery",
        "Birth_weight_in_grams",
        "Blood Loss",
        "Labour min",
    ]

    missing_data = []

    denominator = len(filtered_df)

    for column in quality_columns:

        missing = filtered_df[
            column
        ].isna().sum()

        missing_percentage = (
            missing / denominator * 100
            if denominator
            else 0
        )

        missing_data.append(
            {
                "Variable": column,
                "Missing_Count": missing,
                "Missing_Percent": round(
                    missing_percentage,
                    2
                ),
            }
        )

    missing_df = pd.DataFrame(
        missing_data
    )

    check_count = (
        filtered_df[
            "Data_Quality_Status"
        ] == "CHECK"
    ).sum()

    duplicate_count = (
        filtered_df
        .duplicated(
            keep=False
        )
        .sum()
    )

    complete_count = (
        filtered_df[
            "Data_Quality_Status"
        ] == "OK"
    ).sum()

    st.subheader("📌 Data Quality Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "OK Records",
        f"{complete_count:,}"
    )

    c3.metric(
        "CHECK Records",
        f"{check_count:,}"
    )

    c4.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # MISSINGNESS
    # -------------------------------------------------------------------------

    fig = px.bar(
        missing_df.sort_values(
            "Missing_Percent",
            ascending=False
        ),
        x="Missing_Percent",
        y="Variable",
        orientation="h",
        text="Missing_Percent",
        title="Data Missingness by Variable",
    )

    fig.update_xaxes(
        title="Missing (%)"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    st.plotly_chart(
        chart_layout(fig, 520),
        use_container_width=True
    )

    st.subheader("📋 Missing Data Table")

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # CHECK RECORDS
    # -------------------------------------------------------------------------

    st.subheader("⚠️ Records Marked CHECK")

    st.caption(
        "CHECK records should be verified against the original register rather "
        "than automatically deleted."
    )

    check_df = filtered_df[
        filtered_df[
            "Data_Quality_Status"
        ] == "CHECK"
    ]

    if check_df.empty:

        st.success(
            "No CHECK records are present under the current filters."
        )

    else:

        st.dataframe(
            check_df,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "The full dataset audit identified 25 records requiring quality review "
        "and six exact duplicate groups involving 12 records. These should be "
        "checked against the source register before final cleaning decisions."
    )


# =============================================================================
# PAGE 7 — STATISTICAL EVIDENCE
# =============================================================================

elif page == "📊 Statistical Evidence":

    page_header(
        "📊 Statistical Evidence",
        "Hypothesis-testing results from the full maternal-health analysis dataset.",
    )

    st.info(
        "These statistical results are based on the full analysis dataset and "
        "are not changed by the sidebar filters. Statistical significance indicates "
        "evidence of association, not causation."
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "LBW Tests",
        "5"
    )

    c2.metric(
        "Delivery Tests",
        "5"
    )

    c3.metric(
        "LBW Significant",
        "2 / 5"
    )

    c4.metric(
        "Delivery Significant",
        "3 / 5"
    )

    st.divider()

    tab_lbw, tab_delivery = st.tabs(
        [
            "👶 Low Birth Weight",
            "🚼 Delivery Mode",
        ]
    )

    # -------------------------------------------------------------------------
    # LBW TESTS
    # -------------------------------------------------------------------------

    with tab_lbw:

        st.subheader(
            "👶 Low Birth Weight Tests"
        )

        lbw_tests = pd.DataFrame(
            {
                "Analysis": [
                    "Gestation vs LBW",
                    "Delivery Mode vs LBW",
                    "Age Group vs LBW",
                    "ANC Group vs LBW",
                    "Deformity vs LBW",
                ],
                "Chi_Square": [
                    27.5972,
                    36.6344,
                    5.4333,
                    5.8332,
                    1.5055,
                ],
                "p_value": [
                    4.41235e-06,
                    5.49841e-08,
                    0.142682,
                    0.120016,
                    0.21982,
                ],
            }
        )

        lbw_tests[
            "Interpretation"
        ] = np.where(
            lbw_tests["p_value"] < 0.05,
            "Statistically significant association",
            "Not statistically significant at 0.05",
        )

        st.dataframe(
            lbw_tests,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "Significant associations were observed for gestation vs LBW "
            "and delivery mode vs LBW."
        )

    # -------------------------------------------------------------------------
    # DELIVERY TESTS
    # -------------------------------------------------------------------------

    with tab_delivery:

        st.subheader(
            "🚼 Delivery Mode Tests"
        )

        delivery_tests = pd.DataFrame(
            {
                "Analysis": [
                    "Diagnosis vs Delivery",
                    "ANC Group vs Delivery",
                    "Gestation vs Delivery",
                    "Age Group vs Delivery",
                    "Deformity vs Delivery",
                ],
                "Chi_Square": [
                    260.7371,
                    14.8512,
                    8.1996,
                    5.2151,
                    0.1730,
                ],
                "p_value": [
                    2.98926e-50,
                    0.00194833,
                    0.042061,
                    0.156707,
                    0.677482,
                ],
            }
        )

        delivery_tests[
            "Interpretation"
        ] = np.where(
            delivery_tests["p_value"] < 0.05,
            "Statistically significant association",
            "Not statistically significant at 0.05",
        )

        st.dataframe(
            delivery_tests,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "Diagnosis, ANC group and gestation category showed statistically "
            "significant associations with delivery mode."
        )

        st.warning(
            "Some diagnosis categories contain relatively few records. "
            "Their individual patterns should therefore be interpreted cautiously."
        )


# =============================================================================
# PAGE 8 — AI / ML DECISION SUPPORT
# =============================================================================

elif page == "🤖 AI — Delivery Prediction":

    page_header(
        "🤖 AI — Delivery Prediction",
        "Exploratory machine-learning predictions for Normal versus Caesarean delivery.",
    )

    st.warning(
        "Research decision-support only. These predictions are not clinical "
        "diagnoses and should not replace assessment by qualified healthcare "
        "professionals."
    )

    # -------------------------------------------------------------------------
    # MODEL INFORMATION
    # -------------------------------------------------------------------------

    st.subheader("🧠 AI Component")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Model Inputs")

        st.write(
            "- Maternal age"
        )

        st.write(
            "- ANC visits"
        )

        st.write(
            "- Previous births"
        )

        st.write(
            "- Previous losses"
        )

        st.write(
            "- Gravidae"
        )

        st.write(
            "- Diagnosis"
        )

        st.write(
            "- Gestational age"
        )

    with col2:

        st.markdown("### Trained Models")

        st.write(
            "- Logistic Regression"
        )

        st.write(
            "- Random Forest"
        )

        st.write(
            "- HistGradientBoosting"
        )

        st.write(
            "**Prediction target:**"
        )

        st.write(
            "Normal vs Caesarean delivery"
        )

    st.divider()

    # -------------------------------------------------------------------------
    # INPUT FORM
    # -------------------------------------------------------------------------

    st.subheader(
        "🧑‍⚕️ Delivery Prediction Input"
    )

    st.caption(
        "Enter the maternal and pregnancy information to generate model outputs."
    )

    with st.form(
        "delivery_prediction_form"
    ):

        input_col1, input_col2 = st.columns(2)

        with input_col1:

            age = st.number_input(
                "Maternal Age (years)",
                min_value=10,
                max_value=60,
                value=22,
                step=1,
            )

            anc = st.number_input(
                "ANC Visits",
                min_value=0,
                max_value=20,
                value=3,
                step=1,
            )

            previous_births = st.number_input(
                "Previous Births",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            previous_losses = st.number_input(
                "Previous Losses",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

        with input_col2:

            gravidae = st.number_input(
                "Gravidae",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )

            gestation = st.number_input(
                "Gestation (weeks)",
                min_value=20.0,
                max_value=45.0,
                value=38.0,
                step=0.1,
            )

            diagnosis_options = sorted(
                df[
                    "Diagnosis"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            diagnosis = st.selectbox(
                "Diagnosis",
                diagnosis_options,
            )

        submitted = st.form_submit_button(
            "🚀 Run Delivery Prediction",
            type="primary",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # RUN MODELS
    # -------------------------------------------------------------------------

    if submitted:

        input_data = pd.DataFrame(
            {
                "AGE": [age],
                "ANC": [anc],
                "Previous_Births": [
                    previous_births
                ],
                "Previous_Losses": [
                    previous_losses
                ],
                "Gravidae": [gravidae],
                "Diagnosis": [diagnosis],
                "Gestation": [gestation],
            }
        )

        class_names = {
            0: "Normal",
            1: "Caesarean",
        }

        # ---------------------------------------------------------------------
        # PREDICTIONS
        # ---------------------------------------------------------------------

        logistic_prediction = int(
            delivery_logistic
            .predict(input_data)[0]
        )

        rf_prediction = int(
            delivery_rf
            .predict(input_data)[0]
        )

        hgb_prediction = int(
            delivery_hgb
            .predict(input_data)[0]
        )

        logistic_probability = (
            delivery_logistic
            .predict_proba(input_data)[0]
        )

        rf_probability = (
            delivery_rf
            .predict_proba(input_data)[0]
        )

        hgb_probability = (
            delivery_hgb
            .predict_proba(input_data)[0]
        )

        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        results = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "Random Forest",
                    "HistGradientBoosting",
                ],
                "Prediction": [
                    class_names[
                        logistic_prediction
                    ],
                    class_names[
                        rf_prediction
                    ],
                    class_names[
                        hgb_prediction
                    ],
                ],
                "Caesarean Probability (%)": [
                    logistic_probability[1] * 100,
                    rf_probability[1] * 100,
                    hgb_probability[1] * 100,
                ],
                "Normal Probability (%)": [
                    logistic_probability[0] * 100,
                    rf_probability[0] * 100,
                    hgb_probability[0] * 100,
                ],
            }
        )

        results[
            "Caesarean Probability (%)"
        ] = results[
            "Caesarean Probability (%)"
        ].round(2)

        results[
            "Normal Probability (%)"
        ] = results[
            "Normal Probability (%)"
        ].round(2)

        # ---------------------------------------------------------------------
        # MODEL CARDS
        # ---------------------------------------------------------------------

        st.divider()

        st.subheader(
            "📋 Prediction Results"
        )

        model_col1, model_col2, model_col3 = st.columns(3)

        with model_col1:

            st.metric(
                "Logistic Regression",
                class_names[
                    logistic_prediction
                ],
            )

            st.write(
                f"Caesarean probability: "
                f"**{logistic_probability[1] * 100:.2f}%**"
            )

            st.write(
                f"Normal probability: "
                f"**{logistic_probability[0] * 100:.2f}%**"
            )

        with model_col2:

            st.metric(
                "Random Forest",
                class_names[
                    rf_prediction
                ],
            )

            st.write(
                f"Caesarean probability: "
                f"**{rf_probability[1] * 100:.2f}%**"
            )

            st.write(
                f"Normal probability: "
                f"**{rf_probability[0] * 100:.2f}%**"
            )

        with model_col3:

            st.metric(
                "HistGradientBoosting",
                class_names[
                    hgb_prediction
                ],
            )

            st.write(
                f"Caesarean probability: "
                f"**{hgb_probability[1] * 100:.2f}%**"
            )

            st.write(
                f"Normal probability: "
                f"**{hgb_probability[0] * 100:.2f}%**"
            )

        # ---------------------------------------------------------------------
        # TABLE
        # ---------------------------------------------------------------------

        st.subheader(
            "📊 Detailed Model Output"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------------------
        # MODEL AGREEMENT
        # ---------------------------------------------------------------------

        predictions = [
            logistic_prediction,
            rf_prediction,
            hgb_prediction,
        ]

        normal_votes = predictions.count(0)

        caesarean_votes = predictions.count(1)

        st.subheader(
            "🤝 Model Agreement"
        )

        agreement_col1, agreement_col2 = st.columns(2)

        with agreement_col1:

            st.metric(
                "Normal Votes",
                f"{normal_votes} / 3"
            )

        with agreement_col2:

            st.metric(
                "Caesarean Votes",
                f"{caesarean_votes} / 3"
            )

        if normal_votes > caesarean_votes:

            st.success(
                f"{normal_votes} of the 3 models predicted Normal delivery."
            )

        elif caesarean_votes > normal_votes:

            st.info(
                f"{caesarean_votes} of the 3 models predicted Caesarean delivery."
            )

        else:

            st.warning(
                "The models produced an equal split."
            )

        # ---------------------------------------------------------------------
        # PROBABILITY CHART
        # ---------------------------------------------------------------------

        st.subheader(
            "📈 Predicted Delivery Probabilities"
        )

        probability_chart = results[
            [
                "Model",
                "Normal Probability (%)",
                "Caesarean Probability (%)",
            ]
        ].melt(
            id_vars="Model",
            var_name="Delivery Type",
            value_name="Probability",
        )

        fig = px.bar(
            probability_chart,
            x="Model",
            y="Probability",
            color="Delivery Type",
            barmode="group",
            title="Model Probability Comparison",
        )

        fig.update_yaxes(
            title="Probability (%)",
            range=[0, 100],
        )

        st.plotly_chart(
            chart_layout(fig, 450),
            use_container_width=True
        )

        # ---------------------------------------------------------------------
        # INPUT SUMMARY
        # ---------------------------------------------------------------------

        st.subheader(
            "📝 Prediction Input Summary"
        )

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------------------
        # MODEL NOTES
        # ---------------------------------------------------------------------

        st.subheader(
            "ℹ️ Model Notes"
        )

        st.info(
            "Birth weight, blood loss, labour duration and deformity are excluded "
            "from the pre-delivery prediction models. The trained pipelines use "
            "AGE, ANC, Previous_Births, Previous_Losses, Gravidae, Diagnosis and "
            "Gestation."
        )

        st.warning(
            "The displayed probabilities are model outputs, not established "
            "clinical risk probabilities. Further validation on larger and "
            "independent datasets would be required before operational clinical use."
        )

    else:

        st.info(
            "Enter the required information and click "
            "'🚀 Run Delivery Prediction' to generate the three model outputs."
        )


# =============================================================================
# PAGE 9 — AI BABY RISK PREDICTION
# =============================================================================


def parse_parity(value):
    if pd.isna(value):
        return np.nan, np.nan
    value = str(value).strip()
    match = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", value)
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan


@st.cache_data
def prepare_baby_risk_data(data):
    work = data.copy()

    for column in ["AGE", "ANC", "Gravidae", "Gestation", "Birth_weight_in_grams"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "Parity" in work.columns:
        parsed = work["Parity"].apply(parse_parity)
        work["Previous_Births"] = parsed.apply(lambda x: x[0])
        work["Previous_Losses"] = parsed.apply(lambda x: x[1])
    else:
        work["Previous_Births"] = np.nan
        work["Previous_Losses"] = np.nan

    work["Diagnosis"] = work["Diagnosis"].apply(
        lambda x: str(x).upper().strip() if pd.notna(x) else np.nan
    )
    work["Diagnosis"] = work["Diagnosis"].replace(["", "NAN", "NONE", "<NA>"], np.nan)

    work["Low_Birth_Weight_Model"] = np.nan
    valid_weight = work["Birth_weight_in_grams"].notna()
    work.loc[valid_weight, "Low_Birth_Weight_Model"] = (
        work.loc[valid_weight, "Birth_weight_in_grams"] < 2500
    ).astype(int)

    work["Deformity_Model"] = np.nan
    deformity = work["Deformity"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else np.nan
    )
    work.loc[deformity.isin(["YES", "Y", "1", "TRUE"]), "Deformity_Model"] = 1
    work.loc[deformity.isin(["NO", "N", "0", "FALSE"]), "Deformity_Model"] = 0

    work["Baby_At_Risk"] = np.where(
        (work["Low_Birth_Weight_Model"] == 1) | (work["Deformity_Model"] == 1),
        1,
        np.where(
            (work["Low_Birth_Weight_Model"] == 0) & (work["Deformity_Model"] == 0),
            0,
            np.nan,
        ),
    )
    return work


@st.cache_resource
def train_baby_risk_models(data):
    feature_columns = [
        "AGE", "ANC", "Previous_Births", "Previous_Losses",
        "Gravidae", "Diagnosis", "Gestation",
    ]
    numerical_features = [
        "AGE", "ANC", "Previous_Births", "Previous_Losses", "Gravidae", "Gestation"
    ]
    categorical_features = ["Diagnosis"]

    model_data = data[feature_columns + ["Baby_At_Risk"]].dropna(subset=["Baby_At_Risk"]).copy()
    if len(model_data) < 100:
        return None

    X = model_data[feature_columns]
    y = model_data["Baby_At_Risk"].astype(int)
    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])

    logistic.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    evaluation = []
    for name, model in [("Logistic Regression", logistic), ("Random Forest", rf)]:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0
        evaluation.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall / Sensitivity": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "Specificity": specificity,
            "ROC-AUC": roc_auc_score(y_test, prob),
        })

    return {
        "logistic": logistic,
        "random_forest": rf,
        "evaluation": pd.DataFrame(evaluation),
        "model_data": model_data,
    }


baby_df = prepare_baby_risk_data(df)
baby_models = train_baby_risk_models(baby_df)


if page == "👶 AI — Baby Risk Prediction":
    page_header(
        "👶 AI — Baby Risk Prediction",
        "Estimates the likelihood of an adverse recorded baby outcome from maternal and pregnancy characteristics.",
    )

    st.info(
        "The research target is Baby At Risk = low birth weight (<2,500 g) OR recorded birth deformity. "
        "Birth weight and deformity are outcomes, not prediction inputs."
    )
    st.warning(
        "Research decision-support only. This is not a clinically validated risk score or diagnosis."
    )

    if baby_models is None:
        st.error("The baby-risk model could not be trained from the available outcome data.")
    else:
        model_data = baby_models["model_data"]
        risk_cases = int((model_data["Baby_At_Risk"] == 1).sum())
        no_risk_cases = int((model_data["Baby_At_Risk"] == 0).sum())
        risk_rate = percentage(risk_cases, len(model_data))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Usable Records", f"{len(model_data):,}")
        c2.metric("Baby At Risk", f"{risk_cases:,}")
        c3.metric("No Recorded Risk", f"{no_risk_cases:,}")
        c4.metric("Observed Rate", f"{risk_rate:.2f}%")

        st.subheader("🧠 Model Inputs")
        st.write("Maternal age • ANC visits • Previous births • Previous losses • Gravidae • Diagnosis • Gestational age")

        st.subheader("📈 Model Performance")
        perf = baby_models["evaluation"].copy()
        for col in ["Accuracy", "Precision", "Recall / Sensitivity", "F1 Score", "Specificity", "ROC-AUC"]:
            perf[col] = (perf[col] * 100).round(2)
        st.dataframe(perf, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🧑‍⚕️ Enter Maternal Information")
        diagnosis_options = sorted(baby_df["Diagnosis"].dropna().astype(str).unique().tolist())

        with st.form("baby_risk_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Maternal Age (years)", 10, 60, 22, 1)
                anc = st.number_input("ANC Visits", 0, 20, 3, 1)
                previous_births = st.number_input("Previous Births", 0, 20, 0, 1)
                previous_losses = st.number_input("Previous Losses", 0, 20, 0, 1)
            with col2:
                gravidae = st.number_input("Gravidae", 1, 20, 1, 1)
                gestation = st.number_input("Gestation (weeks)", 20.0, 45.0, 38.0, 0.1)
                diagnosis = st.selectbox("Diagnosis", diagnosis_options)
            submitted = st.form_submit_button("👶 Estimate Baby Risk", type="primary", use_container_width=True)

        if submitted:
            input_data = pd.DataFrame({
                "AGE": [age], "ANC": [anc], "Previous_Births": [previous_births],
                "Previous_Losses": [previous_losses], "Gravidae": [gravidae],
                "Diagnosis": [diagnosis], "Gestation": [gestation],
            })

            logistic_prob = baby_models["logistic"].predict_proba(input_data)[0, 1]
            rf_prob = baby_models["random_forest"].predict_proba(input_data)[0, 1]
            logistic_pred = int(baby_models["logistic"].predict(input_data)[0])
            rf_pred = int(baby_models["random_forest"].predict(input_data)[0])
            pct = logistic_prob * 100
            category = "Lower estimated probability" if pct < 30 else "Intermediate estimated probability" if pct < 60 else "Higher estimated probability"

            st.divider()
            st.subheader("🎯 Estimated Baby Risk")
            r1, r2, r3 = st.columns(3)
            r1.metric("Estimated Probability", f"{pct:.1f}%")
            r2.metric("Primary Model", "Logistic Regression")
            r3.metric("Estimated Category", category)

            st.write(
                f"Based on the entered characteristics, the Logistic Regression model estimates a **{pct:.1f}% model probability** "
                "of the composite adverse baby outcome (low birth weight or recorded deformity)."
            )

            comparison = pd.DataFrame({
                "Model": ["Logistic Regression", "Random Forest"],
                "Baby At Risk Probability (%)": [logistic_prob * 100, rf_prob * 100],
                "Predicted Class": ["At Risk" if logistic_pred else "Not At Risk", "At Risk" if rf_pred else "Not At Risk"],
            })
            comparison["Baby At Risk Probability (%)"] = comparison["Baby At Risk Probability (%)"].round(2)
            st.subheader("🤖 Model Probability Comparison")
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            fig = px.bar(comparison, x="Model", y="Baby At Risk Probability (%)", text="Baby At Risk Probability (%)", title="Estimated Baby-Risk Probability by Model")
            fig.update_yaxes(range=[0, 100], title="Probability (%)")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

            if logistic_pred == rf_pred:
                st.success("Both models produced the same classification: **At Risk**." if logistic_pred else "Both models produced the same classification: **Not At Risk**.")
            else:
                st.warning("The two models produced different classifications, indicating model uncertainty.")

            st.subheader("📝 Maternal Information Used")
            st.dataframe(input_data, use_container_width=True, hide_index=True)

            st.subheader("🔎 Descriptive Screening Flags")
            flags = []
            if age < 18: flags.append("Maternal age below 18")
            if age >= 35: flags.append("Maternal age 35 years or above")
            if anc < 4: flags.append("Fewer than 4 recorded ANC visits")
            if gestation < 37: flags.append("Gestational age below 37 weeks")
            if previous_losses > 0: flags.append("Previous pregnancy loss recorded")
            if flags:
                for flag in flags: st.write(f"• {flag}")
            else:
                st.write("No simple screening flags were triggered by the entered values.")

            st.warning(
                "Important limitation: the register does not contain a dedicated validated maternal-risk label. "
                "Therefore this model estimates P(Baby Adverse Outcome | Recorded Maternal/Pregnancy Characteristics), "
                "not P(Baby Risk | Clinically Diagnosed Maternal Risk)."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "🏥 Nyambene Subcounty Hospital | Maternal Health Analytics & Decision "
    "Support System | Research prototype"
)

st.caption(
    "AI predictions are for research decision-support purposes only."
)