import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="ChurnGuard · Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
    h1, h2, h3, h4, p, label, .stMarkdown { color: #e6edf3 !important; }

    [data-testid="metric-container"] {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 1.25rem 1.5rem; transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover { border-color: #58a6ff; }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important; font-size: 0.78rem !important;
        font-weight: 500 !important; letter-spacing: 0.08em; text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important; font-size: 1.9rem !important;
        font-weight: 600 !important; font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

    .stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
        background-color: #0d1117 !important; border-color: #30363d !important; color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] label {
        color: #8b949e !important; font-size: 0.8rem !important;
        font-weight: 500 !important; letter-spacing: 0.05em; text-transform: uppercase;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd); color: #ffffff;
        border: none; border-radius: 8px; padding: 0.65rem 1.5rem;
        font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600;
        letter-spacing: 0.02em; width: 100%; transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 6px 20px rgba(31, 111, 235, 0.45);
    }

    hr { border-color: #21262d !important; }
    .js-plotly-plot { border-radius: 12px; }

    .section-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #58a6ff !important; margin-bottom: 1rem;
    }

    .risk-card { border-radius: 14px; padding: 1.75rem; margin-bottom: 1.25rem; }
    .risk-high {
        background: linear-gradient(135deg, rgba(248,81,73,0.12), rgba(248,81,73,0.04));
        border: 1px solid rgba(248,81,73,0.35);
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(210,153,34,0.12), rgba(210,153,34,0.04));
        border: 1px solid rgba(210,153,34,0.35);
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(35,134,54,0.12), rgba(35,134,54,0.04));
        border: 1px solid rgba(35,134,54,0.35);
    }
    .risk-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.25rem; }
    .risk-subtitle { font-size: 0.85rem; color: #8b949e; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border: 1px solid #21262d; border-radius: 8px;
        color: #8b949e; font-size: 0.82rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1f2937 !important; border-color: #58a6ff !important; color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA & TRAIN ALL MODELS ---------------- #

@st.cache_resource
def load_and_train():
    df = pd.read_csv("churn_data.csv")
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols     = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # 80/20 stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost (GBM)":       GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    trained  = {}
    metrics  = {}
    roc_data = {}

    for name, clf in models_def.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        trained[name] = pipe
        metrics[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        }
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr.tolist(), tpr.tolist())

    # Feature importance from Random Forest
    rf_pipe      = trained["Random Forest"]
    ohe_features = rf_pipe.named_steps["preprocessor"] \
                       .named_transformers_["cat"] \
                       .get_feature_names_out(categorical_cols).tolist()
    all_features = numeric_cols + ohe_features
    importances  = rf_pipe.named_steps["classifier"].feature_importances_

    feat_df = (
        pd.DataFrame({"feature": all_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    best_model_name = max(metrics, key=lambda k: metrics[k]["ROC-AUC"])
    return trained, metrics, roc_data, feat_df, best_model_name, df

trained_models, all_metrics, roc_data, feat_df, best_model_name, df = load_and_train()

total_customers = len(df)
churn_rate      = df["Churn"].mean()
avg_monthly     = df["MonthlyCharges"].mean()
at_risk_count   = int(total_customers * churn_rate)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-size: 1.3rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.01em;'>🛡️ ChurnGuard</div>
        <div style='font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem;'>Customer Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Prediction Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Choose Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name),
        help="Best model pre-selected by ROC-AUC"
    )
    auc = all_metrics[selected_model]["ROC-AUC"]
    acc = all_metrics[selected_model]["Accuracy"]
    st.markdown(f"""
    <div style='display:flex; gap:0.5rem; margin-bottom:1rem;'>
        <div style='flex:1; background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:0.5rem; text-align:center;'>
            <div style='font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;'>AUC</div>
            <div style='font-size:1rem; font-weight:600; color:#58a6ff; font-family:DM Mono,monospace;'>{auc:.3f}</div>
        </div>
        <div style='flex:1; background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:0.5rem; text-align:center;'>
            <div style='font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;'>Acc</div>
            <div style='font-size:1rem; font-weight:600; color:#3fb950; font-family:DM Mono,monospace;'>{acc:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",         ["Female", "Male"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Partner",        ["No", "Yes"])
    dependents = st.selectbox("Dependents",     ["No", "Yes"])

    st.markdown("---")
    st.markdown('<div class="section-label">Subscription</div>', unsafe_allow_html=True)
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    contract         = st.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])

    st.markdown("---")
    st.markdown('<div class="section-label">Add-ons</div>', unsafe_allow_html=True)
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, max_value=10000.0,
                                      value=float(monthly_charges * tenure) if tenure > 0 else 1000.0, step=10.0)
    payment_method  = st.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("---")
    predict_btn = st.button("⚡  Run Churn Analysis", use_container_width=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: #e6edf3 !important;'>
        Customer Retention Dashboard
    </h1>
    <p style='color: #8b949e; margin: 0.35rem 0 0; font-size: 0.92rem;'>
        Predict churn risk · Compare models · Understand what drives churn
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI ROW ---------------- #

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Customers", f"{total_customers:,}")
with k2:
    st.metric("Dataset Churn Rate", f"{churn_rate*100:.1f}%", delta=f"-{churn_rate*100:.1f}% target", delta_color="inverse")
with k3:
    st.metric("At-Risk Customers", f"{at_risk_count:,}", delta="Needs attention", delta_color="inverse")
with k4:
    best_auc = all_metrics[best_model_name]["ROC-AUC"]
    st.metric("Best Model AUC", f"{best_auc:.3f}", delta=best_model_name)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- BUILD INPUT ---------------- #

def build_input():
    return pd.DataFrame({
        "gender":           [gender],
        "SeniorCitizen":    [1 if senior == "Yes" else 0],
        "Partner":          [partner],
        "Dependents":       [dependents],
        "tenure":           [tenure],
        "PhoneService":     [phone_service],
        "MultipleLines":    ["No"],
        "InternetService":  [internet_service],
        "OnlineSecurity":   [online_security],
        "OnlineBackup":     [online_backup],
        "DeviceProtection": ["No"],
        "TechSupport":      [tech_support],
        "StreamingTV":      [streaming_tv],
        "StreamingMovies":  ["No"],
        "Contract":         [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod":    [payment_method],
        "MonthlyCharges":   [monthly_charges],
        "TotalCharges":     [total_charges],
    })

# ---------------- MAIN LAYOUT ---------------- #

left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ── LEFT: Prediction ─────────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">Churn Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data  = build_input()
        probability = trained_models[selected_model].predict_proba(input_data)[0][1]

        if probability > 0.7:
            risk_level, risk_class, risk_icon = "High Risk",     "risk-high",   "🔴"
            risk_color = "#f85149"
            advice = "Immediate intervention recommended. Offer retention incentives or dedicated account management."
        elif probability > 0.4:
            risk_level, risk_class, risk_icon = "Moderate Risk", "risk-medium", "🟡"
            risk_color = "#d29922"
            advice = "Monitor closely. Consider proactive outreach or upgrading service features."
        else:
            risk_level, risk_class, risk_icon = "Low Risk",      "risk-low",    "🟢"
            risk_color = "#3fb950"
            advice = "Customer appears stable. Focus on deepening engagement and upsell opportunities."

        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;'>
                <span style='font-size:1.4rem;'>{risk_icon}</span>
                <div>
                    <div class="risk-title" style='color:{risk_color};'>{risk_level}</div>
                    <div class="risk-subtitle">Churn Probability: {probability*100:.1f}% · {selected_model}</div>
                </div>
            </div>
            <div style='font-size:0.82rem; color:#8b949e; line-height:1.5;'>{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3", "family": "DM Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#30363d", "tickfont": {"color": "#8b949e", "size": 10}},
                "bar":  {"color": risk_color, "thickness": 0.22},
                "bgcolor": "#161b22", "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(63,185,80,0.12)"},
                    {"range": [40, 70], "color": "rgba(210,153,34,0.12)"},
                    {"range": [70,100], "color": "rgba(248,81,73,0.12)"},
                ],
                "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.85, "value": probability * 100}
            }
        ))
        gauge.update_layout(height=220, margin=dict(l=20,r=20,t=20,b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"family": "DM Sans"})
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label" style="margin-top:1rem;">Key Risk Signals</div>', unsafe_allow_html=True)
        factors = []
        if contract == "Month-to-month":      factors.append(("⚠️ Month-to-month contract",                    "high"))
        if tenure < 12:                        factors.append((f"⚠️ Short tenure ({tenure}mo)",                 "high"))
        if internet_service == "Fiber optic": factors.append(("⚠️ Fiber optic service",                       "medium"))
        if monthly_charges > 80:              factors.append((f"⚠️ High monthly spend (${monthly_charges:.0f})", "medium"))
        if online_security == "No":           factors.append(("ℹ️ No online security",                        "low"))
        if not factors:                        factors.append(("✅ No major risk signals",                     "positive"))

        color_map = {"high": "#f85149", "medium": "#d29922", "low": "#8b949e", "positive": "#3fb950"}
        for label, level in factors[:4]:
            st.markdown(f"""
            <div style='padding:0.5rem 0.75rem; border-radius:6px; background:rgba(255,255,255,0.03);
                        border-left:3px solid {color_map[level]}; margin-bottom:0.4rem;
                        font-size:0.82rem; color:#c9d1d9;'>{label}</div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='border:1px dashed #30363d; border-radius:14px; padding:2.5rem 1.5rem;
                    text-align:center; margin-bottom:1rem;'>
            <div style='font-size:2rem; margin-bottom:0.75rem;'>⚡</div>
            <div style='color:#8b949e; font-size:0.88rem; line-height:1.6;'>
                Configure customer details in the sidebar,<br>
                then click <strong style='color:#58a6ff;'>Run Churn Analysis</strong> to see results.
            </div>
        </div>
        """, unsafe_allow_html=True)

        gauge_empty = go.Figure(go.Indicator(
            mode="gauge+number", value=0,
            number={"suffix": "%", "font": {"size": 28, "color": "#30363d", "family": "DM Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#21262d", "tickfont": {"color": "#30363d", "size": 10}},
                "bar":  {"color": "#21262d", "thickness": 0.22},
                "bgcolor": "#161b22", "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(63,185,80,0.05)"},
                    {"range": [40, 70], "color": "rgba(210,153,34,0.05)"},
                    {"range": [70,100], "color": "rgba(248,81,73,0.05)"},
                ],
            }
        ))
        gauge_empty.update_layout(height=220, margin=dict(l=20,r=20,t=20,b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font={"family": "DM Sans"})
        st.plotly_chart(gauge_empty, use_container_width=True, config={"displayModeBar": False})

# ── RIGHT: Insights ───────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">Analytics & Model Intelligence</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Churn by Segment",
        "📈 Tenure Analysis",
        "💰 Revenue Impact",
        "🤖 Model Comparison",
        "🔍 Feature Importance",
    ])

    # ── TAB 1 ────────────────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
            contract_churn.columns = ["Contract", "Churn Rate"]
            contract_churn["Churn Rate"] *= 100
            fig1 = go.Figure(go.Bar(
                x=contract_churn["Churn Rate"], y=contract_churn["Contract"], orientation="h",
                marker=dict(color=contract_churn["Churn Rate"],
                            colorscale=[[0,"#1f6feb"],[0.5,"#d29922"],[1,"#f85149"]], line=dict(width=0)),
                text=[f"{v:.1f}%" for v in contract_churn["Churn Rate"]],
                textposition="outside", textfont=dict(color="#8b949e", size=11)
            ))
            fig1.update_layout(title=dict(text="By Contract Type", font=dict(color="#8b949e", size=12), x=0),
                               height=200, margin=dict(l=10,r=40,t=35,b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(showgrid=False, showticklabels=False, color="#30363d"),
                               yaxis=dict(color="#8b949e", tickfont=dict(size=11)), font=dict(family="DM Sans"))
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            isp_churn = df.groupby("InternetService")["Churn"].mean().reset_index()
            isp_churn.columns = ["Service", "Churn Rate"]
            isp_churn["Churn Rate"] *= 100
            colors_isp = {"DSL": "#1f6feb", "Fiber optic": "#f85149", "No": "#3fb950"}
            fig2 = go.Figure(go.Bar(
                x=isp_churn["Service"], y=isp_churn["Churn Rate"],
                marker=dict(color=[colors_isp.get(s, "#58a6ff") for s in isp_churn["Service"]], line=dict(width=0)),
                text=[f"{v:.1f}%" for v in isp_churn["Churn Rate"]],
                textposition="outside", textfont=dict(color="#8b949e", size=11)
            ))
            fig2.update_layout(title=dict(text="By Internet Service", font=dict(color="#8b949e", size=12), x=0),
                               height=200, margin=dict(l=10,r=10,t=35,b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(color="#8b949e", tickfont=dict(size=11), showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor="#21262d", color="#30363d", showticklabels=False),
                               font=dict(family="DM Sans"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        pay_churn = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        pay_churn.columns = ["Method", "Churn Rate"]
        pay_churn["Churn Rate"] *= 100
        pay_churn = pay_churn.sort_values("Churn Rate", ascending=True)
        fig3 = go.Figure(go.Bar(
            x=pay_churn["Churn Rate"], y=pay_churn["Method"], orientation="h",
            marker=dict(color=pay_churn["Churn Rate"],
                        colorscale=[[0,"#1f6feb"],[1,"#f85149"]], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in pay_churn["Churn Rate"]],
            textposition="outside", textfont=dict(color="#8b949e", size=11)
        ))
        fig3.update_layout(title=dict(text="Churn Rate by Payment Method", font=dict(color="#8b949e", size=12), x=0),
                           height=200, margin=dict(l=10,r=60,t=35,b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(showgrid=False, showticklabels=False, color="#30363d"),
                           yaxis=dict(color="#8b949e", tickfont=dict(size=10)), font=dict(family="DM Sans"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 2 ────────────────────────────────────────────────────────────────
    with tab2:
        df_churned  = df[df["Churn"] == 1]["tenure"]
        df_retained = df[df["Churn"] == 0]["tenure"]
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=df_churned,  name="Churned",
                                    marker_color="rgba(248,81,73,0.7)",  xbins=dict(size=3), opacity=0.85))
        fig4.add_trace(go.Histogram(x=df_retained, name="Retained",
                                    marker_color="rgba(31,111,235,0.6)", xbins=dict(size=3), opacity=0.75))
        fig4.update_layout(title=dict(text="Tenure Distribution: Churned vs Retained", font=dict(color="#8b949e", size=12), x=0),
                           barmode="overlay", height=230, margin=dict(l=10,r=10,t=35,b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Tenure (months)", color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
                           yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
                           legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)"),
                           font=dict(family="DM Sans"))
        fig4.add_vline(x=tenure, line_color="#58a6ff", line_width=2, line_dash="dot",
                       annotation_text=f"This customer ({tenure}mo)",
                       annotation_font=dict(color="#58a6ff", size=10),
                       annotation_position="top right")
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        df_tmp = df.copy()
        df_tmp["tenure_bucket"] = pd.cut(df_tmp["tenure"], bins=[0,12,24,36,48,72],
                                          labels=["0–12mo","13–24mo","25–36mo","37–48mo","49–72mo"])
        tenure_churn = df_tmp.groupby("tenure_bucket", observed=True)["Churn"].mean().reset_index()
        tenure_churn["Churn Rate"] = tenure_churn["Churn"] * 100
        fig5 = go.Figure(go.Scatter(
            x=tenure_churn["tenure_bucket"].astype(str), y=tenure_churn["Churn Rate"],
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2.5),
            marker=dict(size=8, color="#58a6ff", line=dict(color="#0d1117", width=2)),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
        ))
        fig5.update_layout(title=dict(text="Churn Rate by Tenure Cohort", font=dict(color="#8b949e", size=12), x=0),
                           height=200, margin=dict(l=10,r=10,t=35,b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                           yaxis=dict(color="#8b949e", gridcolor="#21262d", ticksuffix="%"),
                           font=dict(family="DM Sans"))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 3 ────────────────────────────────────────────────────────────────
    with tab3:
        sample = df.sample(min(600, len(df)), random_state=42)
        fig6 = go.Figure()
        for churn_val, lbl, clr in [(0,"Retained","#1f6feb"),(1,"Churned","#f85149")]:
            subset = sample[sample["Churn"] == churn_val]
            fig6.add_trace(go.Scatter(x=subset["tenure"], y=subset["MonthlyCharges"],
                                      mode="markers", name=lbl,
                                      marker=dict(color=clr, size=5, opacity=0.6, line=dict(width=0))))
        fig6.add_trace(go.Scatter(x=[tenure], y=[monthly_charges], mode="markers", name="This Customer",
                                  marker=dict(color="#58a6ff", size=14, symbol="star",
                                              line=dict(color="#ffffff", width=1.5))))
        fig6.update_layout(title=dict(text="Monthly Charges vs Tenure (Sample)", font=dict(color="#8b949e", size=12), x=0),
                           height=260, margin=dict(l=10,r=10,t=35,b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Tenure (months)", color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
                           yaxis=dict(title="Monthly Charges ($)", color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
                           legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)"),
                           font=dict(family="DM Sans"))
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

        avg_charges_churned = df[df["Churn"] == 1]["MonthlyCharges"].mean()
        revenue_at_risk     = avg_charges_churned * at_risk_count
        ltv_diff = df.groupby("Churn")["TotalCharges"].mean()
        ltv_gap  = ltv_diff[0] - ltv_diff[1]
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Monthly Revenue at Risk",           f"${revenue_at_risk:,.0f}", delta="Based on churn cohort", delta_color="inverse")
        with r2:
            st.metric("Avg LTV Gap (Retained vs Churned)", f"${ltv_gap:,.0f}",         delta="Retention value",       delta_color="normal")

    # ── TAB 4: Model Comparison ───────────────────────────────────────────────
    with tab4:
        metric_names = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1 Score"]
        model_colors = {
            "Logistic Regression": "#58a6ff",
            "Random Forest":       "#3fb950",
            "XGBoost (GBM)":       "#f0883e",
        }

        st.markdown('<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.75rem;">Evaluated on 20% held-out test set (stratified) · Best model by ROC-AUC is highlighted</div>', unsafe_allow_html=True)

        m_cols = st.columns(3)
        for i, (mname, mvals) in enumerate(all_metrics.items()):
            is_best = mname == best_model_name
            border  = "#58a6ff" if is_best else "#21262d"
            badge   = " 🏆" if is_best else ""
            with m_cols[i]:
                rows = "".join([
                    f"<div style='display:flex; justify-content:space-between; padding:0.3rem 0;"
                    f"border-bottom:1px solid #21262d;'>"
                    f"<span style='color:#8b949e; font-size:0.78rem;'>{k}</span>"
                    f"<span style='color:#e6edf3; font-size:0.78rem; font-family:DM Mono,monospace;"
                    f"font-weight:600;'>{mvals[k]:.3f}</span></div>"
                    for k in metric_names
                ])
                st.markdown(f"""
                <div style='background:#161b22; border:1px solid {border}; border-radius:10px;
                            padding:1rem 1.25rem; height:100%;'>
                    <div style='font-size:0.88rem; font-weight:600; color:{model_colors[mname]};
                                margin-bottom:0.75rem;'>{mname}{badge}</div>
                    {rows}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROC curves
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="#30363d", width=1, dash="dot"))
        for mname, (fpr, tpr) in roc_data.items():
            auc_val = all_metrics[mname]["ROC-AUC"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{mname} (AUC={auc_val:.3f})",
                line=dict(color=model_colors[mname], width=2.5)
            ))
        fig_roc.update_layout(
            title=dict(text="ROC Curves — All Models", font=dict(color="#8b949e", size=12), x=0),
            height=300, margin=dict(l=10,r=10,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="False Positive Rate", color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
            yaxis=dict(title="True Positive Rate",  color="#8b949e", gridcolor="#21262d", title_font=dict(size=11)),
            legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)"),
            font=dict(family="DM Sans")
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

        # Grouped bar: metric comparison
        fig_bar = go.Figure()
        for mname in all_metrics:
            fig_bar.add_trace(go.Bar(
                name=mname, x=metric_names,
                y=[all_metrics[mname][m] for m in metric_names],
                marker=dict(color=model_colors[mname], line=dict(width=0)),
                text=[f"{all_metrics[mname][m]:.2f}" for m in metric_names],
                textposition="outside", textfont=dict(color="#8b949e", size=10)
            ))
        fig_bar.update_layout(
            title=dict(text="Metric Comparison Across Models", font=dict(color="#8b949e", size=12), x=0),
            barmode="group", height=280, margin=dict(l=10,r=10,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.12]),
            legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)"),
            font=dict(family="DM Sans")
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 5: Feature Importance ─────────────────────────────────────────────
    with tab5:
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.75rem;">Based on Random Forest · Top 15 features ranked by importance score</div>', unsafe_allow_html=True)

        def clean_name(s):
            for prefix in ["cat__", "num__"]:
                if s.startswith(prefix):
                    s = s[len(prefix):]
            parts = s.split("_")
            return " · ".join(parts[:2]) if len(parts) >= 2 else s

        feat_display = feat_df.copy()
        feat_display["label"] = feat_display["feature"].apply(clean_name)
        feat_display = feat_display.sort_values("importance", ascending=True)

        max_imp    = feat_display["importance"].max()
        bar_colors = [
            "#f85149" if v >= max_imp * 0.75 else
            "#d29922" if v >= max_imp * 0.45 else
            "#58a6ff"
            for v in feat_display["importance"]
        ]

        fig_fi = go.Figure(go.Bar(
            x=feat_display["importance"], y=feat_display["label"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in feat_display["importance"]],
            textposition="outside", textfont=dict(color="#8b949e", size=10)
        ))
        fig_fi.update_layout(
            title=dict(text="Top 15 Churn Drivers (Random Forest)", font=dict(color="#8b949e", size=12), x=0),
            height=460, margin=dict(l=10,r=60,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#21262d", showticklabels=False, color="#30363d"),
            yaxis=dict(color="#8b949e", tickfont=dict(size=11)),
            font=dict(family="DM Sans")
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div style='display:flex; gap:1.5rem; font-size:0.78rem; color:#8b949e; margin-top:0.25rem;'>
            <span><span style='color:#f85149;'>■</span> High impact</span>
            <span><span style='color:#d29922;'>■</span> Medium impact</span>
            <span><span style='color:#58a6ff;'>■</span> Lower impact</span>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 callout cards
        top3 = feat_display.sort_values("importance", ascending=False).head(3)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Top Churn Driver Insights</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
            col.markdown(f"""
            <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                        padding:1rem; text-align:center;'>
                <div style='font-size:1.5rem; margin-bottom:0.4rem;'>🔑</div>
                <div style='font-size:0.82rem; font-weight:600; color:#e6edf3; margin-bottom:0.3rem;'>{row['label']}</div>
                <div style='font-size:0.75rem; color:#8b949e;'>Importance:
                    <span style='color:#58a6ff; font-family:DM Mono,monospace;'>{row['importance']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='border-top:1px solid #21262d; padding-top:1rem; display:flex; justify-content:space-between;
            font-size:0.75rem; color:#484f58;'>
    <span>🛡️ ChurnGuard Analytics · 80/20 Train-Test Split · 3-Model Comparison</span>
    <span>Best model: {best_model_name} · AUC {all_metrics[best_model_name]['ROC-AUC']:.3f}</span>
</div>
""", unsafe_allow_html=True)