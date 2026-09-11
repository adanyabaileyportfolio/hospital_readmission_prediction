from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_PATH = PROJECT_ROOT / "reports" / "tables"

# color palette
PRIMARY = "#00205B"
SECONDARY = "#5BC2E7"
ACCENT = "#FFB81C"
BACKGROUND = "#F6F8FB"
CARD_BACKGROUND = "#FFFFFF"
TEXT = "#20242A"
MUTED_TEXT = "#667085"

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


@st.cache_data
def load_table(filename):
    """Load a project results table."""

    return pd.read_csv(TABLES_PATH / filename)


# Load project results
target_distribution = load_table("target_distribution.csv")
age_results = load_table("readmission_by_age.csv")
race_results = load_table("readmission_by_race.csv")
gender_results = load_table("readmission_by_gender.csv")
split_summary = load_table("patient_level_split_summary.csv")

model_results = load_table(
    "model_evaluation_default_threshold.csv"
)

threshold_results = load_table(
    "model_threshold_comparison.csv"
)

feature_importance = load_table(
    "random_forest_permutation_importance.csv"
)

subgroup_results = load_table(
    "subgroup_performance.csv"
)


# styling
st.markdown(
    """
    <style>
    /* Page */
    .stApp {
        background-color: #F6F8FB;
        color: #20242A;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Responsive header */
    .main-title {
        color: #00205B;
        font-size: clamp(1.85rem, 4vw, 2.65rem);
        font-weight: 800;
        line-height: 1.18;
        letter-spacing: -0.025rem;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
        max-width: 100%;
        margin: 0 0 0.4rem;
        padding: 0;
    }

    .subtitle {
        color: #667085;
        font-size: clamp(0.95rem, 1.7vw, 1.08rem);
        line-height: 1.5;
        white-space: normal;
        max-width: 100%;
        margin-bottom: 1.4rem;
    }

    /* Disclaimer */
    .disclaimer {
        color: #20242A;
        background-color: #EDF7FB;
        border: 1px solid #C5E6F2;
        border-left: 6px solid #5BC2E7;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 1rem 0 1.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        min-height: 120px;
        background-color: #FFFFFF;
        border: 1px solid #DCE3EC;
        border-top: 4px solid #5BC2E7;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 12px rgba(0, 32, 91, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: #667085;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #00205B;
        font-weight: 750;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #667085;
        border-radius: 12px 12px 0 0;
        padding: 0.65rem 1rem;
        font-weight: 650;
    }

    [data-testid="stTabs"] button:hover {
        color: #00205B;
        background-color: #EDF7FB;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00205B;
        background-color: #E5F5FB;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #00205B;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        color: #FFFFFF;
        background-color: #00205B;
        border: 1px solid #00205B;
        border-radius: 14px;
        padding: 0.6rem 1.2rem;
        font-weight: 650;
        transition: all 0.2s ease;
        box-shadow: 0 3px 8px rgba(0, 32, 91, 0.15);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: #00205B;
        background-color: #E5F5FB;
        border-color: #5BC2E7;
        transform: translateY(-1px);
    }

    /* Dropdowns */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #CBD5E1;
        border-radius: 13px;
    }

    [data-baseweb="select"] > div:focus-within {
        border-color: #5BC2E7;
        box-shadow: 0 0 0 1px #5BC2E7;
    }

    /* Slider */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #00205B;
        border-color: #00205B;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE3EC;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE3EC;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #00205B;
        letter-spacing: -0.01rem;
    }

    hr {
        border-color: #DCE3EC;
    }

    /* Mobile layout */
    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 1.85rem;
        }

        [data-testid="stMetric"] {
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    """
    <div class="main-title">
        Predicting 30-Day Hospital Readmissions
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        An educational machine-learning analysis of hospital encounters
        involving patients with diabetes
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer">
        <strong>Educational demonstration:</strong>
        This application uses historical data from 1999–2008 and is
        not intended for clinical decision-making or use with real patients.
    </div>
    """,
    unsafe_allow_html=True
)


# Navigation tabs
overview_tab, eda_tab, model_tab, audit_tab = st.tabs([
    "Project Overview",
    "Exploratory Analysis",
    "Model Performance",
    "Subgroup Audit"
])


# Project overview
with overview_tab:
    st.header("Project overview")

    total_encounters = int(
        target_distribution["encounters"].sum()
    )

    readmission_rate = float(
        target_distribution.loc[
            target_distribution["outcome"]
            == "Readmitted within 30 days",
            "percentage"
        ].iloc[0]
    )

    unique_patients = int(
        split_summary["unique_patients"].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Eligible encounters",
        f"{total_encounters:,}"
    )

    col2.metric(
        "Unique patients",
        f"{unique_patients:,}"
    )

    col3.metric(
        "30-day readmission rate",
        f"{readmission_rate:.2f}%"
    )

    col4.metric(
        "Hospitals and networks",
        "130"
    )

    st.subheader("Research question")

    st.write(
        "Which patient, hospitalization, and treatment "
        "characteristics are associated with readmission within "
        "30 days of discharge?"
    )

    st.subheader("Project workflow")

    workflow = pd.DataFrame({
        "Stage": [
            "Data preparation",
            "Database analysis",
            "Leakage prevention",
            "Modeling",
            "Evaluation",
            "Responsible audit"
        ],
        "Approach": [
            "Python cleaning and ICD-9 diagnosis grouping",
            "SQLite tables, lookup joins, and analysis queries",
            "Patient-level training and test split",
            (
                "Dummy classifier, logistic regression, "
                "and random forest"
            ),
            (
                "ROC-AUC, PR-AUC, calibration, "
                "and threshold analysis"
            ),
            (
                "Performance comparisons by race, "
                "gender, and age"
            )
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Key descriptive finding")

    st.info(
        "Encounters involving patients with three or more inpatient "
        "visits during the preceding year had a 26.4% observed "
        "30-day readmission rate."
    )

    st.subheader("Dataset")

    st.write(
        "The UCI Diabetes 130-US Hospitals dataset contains inpatient "
        "encounters collected from 130 U.S. hospitals and integrated "
        "delivery networks between 1999 and 2008. After excluding "
        "invalid-gender records and encounters ending in death or "
        "hospice care, 99,340 encounters remained."
    )


# Exploratory analysis
with eda_tab:
    st.header("Exploratory analysis")

    demographic_choice = st.selectbox(
        "Compare readmission rates by:",
        ["Age", "Race", "Gender"]
    )

    demographic_tables = {
        "Age": (
            age_results,
            "age",
            "Age Group"
        ),
        "Race": (
            race_results,
            "race",
            "Recorded Race"
        ),
        "Gender": (
            gender_results,
            "gender",
            "Recorded Gender"
        )
    }

    selected_data, group_column, axis_title = (
        demographic_tables[demographic_choice]
    )

    selected_data = selected_data.copy()

    selected_data[group_column] = (
        selected_data[group_column]
        .fillna("Missing")
        .astype(str)
    )

    demographic_chart = px.bar(
        selected_data,
        x=group_column,
        y="readmission_rate_pct",
        text="readmission_rate_pct",
        color_discrete_sequence=[PRIMARY],
        labels={
            group_column: axis_title,
            "readmission_rate_pct": "Readmission Rate (%)"
        },
        title=(
            f"Observed 30-Day Readmission Rate by "
            f"{demographic_choice}"
        )
    )

    demographic_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=0
    )

    demographic_chart.update_layout(
        showlegend=False,
        yaxis_title="Readmission Rate (%)",
        plot_bgcolor=CARD_BACKGROUND,
        paper_bgcolor=CARD_BACKGROUND,
        title_font_color=PRIMARY
    )

    st.plotly_chart(
        demographic_chart,
        use_container_width=True
    )

    with st.expander("View the underlying summary table"):
        st.dataframe(
            selected_data[
                [
                    group_column,
                    "encounters",
                    "readmissions",
                    "readmission_rate_pct"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Interpretation")

    st.write(
        "These comparisons are unadjusted. Differences may reflect "
        "health status, prior utilization, access to care, hospital "
        "practices, structural inequities, or other measured and "
        "unmeasured factors. They do not demonstrate causation."
    )


# Model performance
with model_tab:
    st.header("Model performance")

    display_metrics = model_results[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "specificity",
            "f1",
            "brier_score"
        ]
    ].copy()

    st.dataframe(
        display_metrics.style.format({
            "roc_auc": "{:.3f}",
            "pr_auc": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "specificity": "{:.3f}",
            "f1": "{:.3f}",
            "brier_score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric_choice = st.selectbox(
        "Choose a model-comparison metric:",
        [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "specificity",
            "f1"
        ]
    )

    metric_chart = px.bar(
        display_metrics,
        x="model",
        y=metric_choice,
        text=metric_choice,
        color="model",
        color_discrete_sequence=[
            MUTED_TEXT,
            SECONDARY,
            PRIMARY
        ],
        labels={
            "model": "Model",
            metric_choice: (
                metric_choice
                .replace("_", " ")
                .upper()
            )
        },
        title=(
            "Model Comparison: "
            f"{metric_choice.replace('_', ' ').upper()}"
        )
    )

    metric_chart.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker_line_width=0
    )

    metric_chart.update_layout(
        showlegend=False,
        yaxis_range=[0, 1],
        plot_bgcolor=CARD_BACKGROUND,
        paper_bgcolor=CARD_BACKGROUND,
        title_font_color=PRIMARY
    )

    st.plotly_chart(
        metric_chart,
        use_container_width=True
    )

    st.subheader("Threshold sensitivity")

    threshold_model = st.selectbox(
        "Model:",
        ["Logistic regression", "Random forest"],
        key="threshold_model"
    )

    available_thresholds = sorted(
        threshold_results.loc[
            threshold_results["model"] == threshold_model,
            "threshold"
        ].unique()
    )

    selected_threshold = st.select_slider(
        "Classification threshold:",
        options=available_thresholds,
        value=0.50
    )

    threshold_row = threshold_results[
        (
            threshold_results["model"] == threshold_model
        )
        & (
            threshold_results["threshold"]
            == selected_threshold
        )
    ].iloc[0]

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Precision",
        f"{threshold_row['precision']:.1%}"
    )

    t2.metric(
        "Recall",
        f"{threshold_row['recall']:.1%}"
    )

    t3.metric(
        "Specificity",
        f"{threshold_row['specificity']:.1%}"
    )

    t4.metric(
        "F1 score",
        f"{threshold_row['f1']:.3f}"
    )

    st.caption(
        "Lower thresholds generally identify more readmissions but "
        "also produce more false-positive classifications. These "
        "values are a sensitivity analysis, not optimized clinical "
        "cutoffs."
    )

    st.subheader("Confusion matrix at the 0.50 threshold")

    confusion_model = st.selectbox(
        "Select a model:",
        model_results["model"].tolist(),
        key="confusion_model"
    )

    confusion_row = model_results[
        model_results["model"] == confusion_model
    ].iloc[0]

    confusion_values = [
        [
            int(confusion_row["true_negatives"]),
            int(confusion_row["false_positives"])
        ],
        [
            int(confusion_row["false_negatives"]),
            int(confusion_row["true_positives"])
        ]
    ]

    confusion_chart = go.Figure(
        data=go.Heatmap(
            z=confusion_values,
            x=[
                "Predicted negative",
                "Predicted positive"
            ],
            y=[
                "Actual negative",
                "Actual positive"
            ],
            colorscale=[
                [0, "#F8EDF0"],
                [0.5, SECONDARY],
                [1, PRIMARY]
            ],
            text=confusion_values,
            texttemplate="%{text:,}",
            showscale=False
        )
    )

    confusion_chart.update_layout(
        title=(
            f"{confusion_model}: "
            "Test-Set Confusion Matrix"
        ),
        xaxis_title="Predicted outcome",
        yaxis_title="Actual outcome",
        plot_bgcolor=CARD_BACKGROUND,
        paper_bgcolor=CARD_BACKGROUND,
        title_font_color=PRIMARY
    )

    st.plotly_chart(
        confusion_chart,
        use_container_width=True
    )

    st.subheader("Random-forest feature importance")

    top_features = (
        feature_importance
        .head(15)
        .sort_values("importance_mean")
    )

    importance_chart = px.bar(
        top_features,
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        color_discrete_sequence=[PRIMARY],
        labels={
            "importance_mean": (
                "Decrease in Average Precision After Permutation"
            ),
            "feature": "Feature"
        }
    )

    importance_chart.update_traces(
        marker_line_width=0
    )

    importance_chart.update_layout(
        showlegend=False,
        plot_bgcolor=CARD_BACKGROUND,
        paper_bgcolor=CARD_BACKGROUND,
        title_font_color=PRIMARY
    )

    st.plotly_chart(
        importance_chart,
        use_container_width=True
    )

    st.info(
        "Previous inpatient utilization was the strongest consistent "
        "predictor. Predictive importance describes model behavior "
        "and does not establish causation."
    )


# Subgroup audit
with audit_tab:
    st.header("Subgroup performance audit")

    audit_model = st.selectbox(
        "Model operating point:",
        [
            "Logistic regression",
            "Random forest"
        ],
        key="audit_model"
    )

    group_choice = st.selectbox(
        "Subgroup variable:",
        ["race", "gender", "age"]
    )

    audit_metric = st.selectbox(
        "Performance metric:",
        [
            "recall",
            "precision",
            "false_positive_rate",
            "false_negative_rate",
            "roc_auc"
        ]
    )

    selected_audit = subgroup_results[
        (
            subgroup_results["model"] == audit_model
        )
        & (
            subgroup_results["group_variable"]
            == group_choice
        )
    ].copy()

    audit_chart = px.bar(
        selected_audit,
        x="group",
        y=audit_metric,
        text=audit_metric,
        color_discrete_sequence=[PRIMARY],
        hover_data=[
            "encounters",
            "readmissions",
            "prevalence_pct"
        ],
        labels={
            "group": group_choice.title(),
            audit_metric: (
                audit_metric
                .replace("_", " ")
                .title()
            )
        },
        title=(
            f"{audit_model}: "
            f"{audit_metric.replace('_', ' ').title()} "
            f"by {group_choice.title()}"
        )
    )

    audit_chart.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        marker_line_width=0
    )

    audit_chart.update_layout(
        showlegend=False,
        yaxis_range=[0, 1],
        plot_bgcolor=CARD_BACKGROUND,
        paper_bgcolor=CARD_BACKGROUND,
        title_font_color=PRIMARY
    )

    st.plotly_chart(
        audit_chart,
        use_container_width=True
    )

    st.warning(
        "Subgroup metrics are descriptive and are less stable for "
        "groups with few encounters or readmissions. Differences do "
        "not by themselves establish that a model is fair or unfair."
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - The dataset is historical and may not reflect current care.
        - Readmission records may not capture every admission outside
          participating systems.
        - Several variables contain missing or broadly coded values.
        - Observed relationships do not establish causation.
        - Model discrimination was moderate rather than strong.
        - Performance varied across age and recorded demographic groups.
        - External validation would be required before any real use.
        """
    )


st.divider()

st.caption(
    "Data source: UCI Diabetes 130-US Hospitals dataset. "
    "Created as an educational healthcare analytics portfolio project."
)