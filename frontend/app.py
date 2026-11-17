import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

API_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
THRESHOLD_DEFAULT = 0.536

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'DM Sans', sans-serif;}

/* Force a consistent light theme, even when the browser/OS is in dark mode */
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#F7F8FA;color:#172033;}
[data-testid="stHeader"] {background:#F7F8FA;}
[data-testid="stHeader"] * {color:#172033 !important;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E7EAF0;}
.block-container {padding-top:1.8rem;max-width:1440px;}

/* Text colours (headings, paragraphs, captions, labels) */
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"] *, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {color:#172033 !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {color:#334155;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color:#64748B !important;}
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {color:#172033 !important;}
[data-testid="stSidebar"] * {color:#172033;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#64748B !important;}
hr {border-color:#E7EAF0 !important;}

/* Inputs: select boxes, text and number inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
    background:#FFFFFF !important;color:#172033 !important;border-color:#D8DEE9 !important;}
[data-baseweb="select"] *, [data-baseweb="input"] * {color:#172033 !important;}
[data-baseweb="popover"] *, [data-baseweb="menu"] * {background:#FFFFFF;color:#172033 !important;}
[data-testid="stNumberInput"] button {background:#F1F5F9 !important;color:#172033 !important;}

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background:#FFFFFF;color:#172033;border:1px solid #D8DEE9;border-radius:10px;font-weight:600;}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color:#2F6B9A;color:#2F6B9A;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background:#2F6B9A !important;color:#FFFFFF !important;border:1px solid #2F6B9A !important;}
button[data-testid="stBaseButton-primaryFormSubmit"] *, button[data-testid="stBaseButton-primary"] * {color:#FFFFFF !important;}

/* Form container, tables, expanders, alerts */
[data-testid="stForm"] {background:#FFFFFF;border:1px solid #E7EAF0;border-radius:13px;padding:1.2rem;}
[data-testid="stExpander"] {background:#FFFFFF;border:1px solid #E7EAF0;border-radius:10px;}
[data-testid="stExpander"] * {color:#172033;}
[data-testid="stDataFrame"] {background:#FFFFFF;border:1px solid #E7EAF0;border-radius:10px;}
[data-testid="stAlert"] {border-radius:10px;}
[data-testid="stAlert"] * {color:#172033 !important;}
[data-testid="stRadio"] label p {color:#172033 !important;}
.metric-card {background:#fff;border:1px solid #E7EAF0;border-radius:13px;padding:16px 18px;min-height:105px;}
.metric-label {color:#64748B;font-size:.78rem;font-weight:700;letter-spacing:.04em;}
.metric-value {color:#172033;font-size:1.6rem;font-weight:700;margin-top:8px;}
.metric-note {color:#94A3B8;font-size:.75rem;margin-top:3px;}
</style>
""", unsafe_allow_html=True)

def metric(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def show_chart(fig):
    """Apply a light, readable style to every Plotly chart, then render it."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, sans-serif", color="#172033"),
        legend=dict(font=dict(color="#172033")),
    )
    fig.update_xaxes(gridcolor="#EEF1F6", linecolor="#D8DEE9", tickfont=dict(color="#334155"),
                     title_font=dict(color="#334155"))
    fig.update_yaxes(gridcolor="#EEF1F6", linecolor="#D8DEE9", tickfont=dict(color="#334155"),
                     title_font=dict(color="#334155"))
    st.plotly_chart(fig, use_container_width=True, theme=None)

def api_url():
    return st.session_state.api_url.rstrip("/")

def normalize_result(data):
    pred = data.get("prediction", data.get("churn_prediction", data.get("label")))
    prob = data.get("churn_probability", data.get("probability", data.get("churn_prob")))
    threshold = data.get("threshold", THRESHOLD_DEFAULT)
    model = data.get("model", "XGBoost")
    if isinstance(pred, (int, float)):
        pred = "Churn" if int(pred) == 1 else "No Churn"
    if isinstance(pred, str):
        p = pred.strip().lower()
        if p in ("1", "yes", "true", "churn", "will churn"): pred = "Churn"
        elif p in ("0", "no", "false", "no churn", "not churn"): pred = "No Churn"
    try:
        prob = float(prob)
        if prob <= 1: prob *= 100
    except (TypeError, ValueError):
        prob = None
    try:
        threshold = float(threshold)
        if threshold <= 1: threshold *= 100
    except (TypeError, ValueError):
        threshold = THRESHOLD_DEFAULT * 100
    return pred, prob, threshold, model

if "api_url" not in st.session_state:
    st.session_state.api_url = API_DEFAULT
if "prediction" not in st.session_state:
    st.session_state.prediction = None

with st.sidebar:
    st.markdown("# ◈ ChurnIQ")
    st.caption("Customer Intelligence Platform")
    page = st.radio("WORKSPACE", ["Overview", "Churn Prediction", "Model Analytics"])
    st.divider()
    st.markdown("**Backend connection**")
    st.session_state.api_url = st.text_input("FastAPI Base URL", st.session_state.api_url).strip().rstrip("/")
    if st.button("Check API connection", use_container_width=True):
        try:
            r = requests.get(f"{api_url()}/health", timeout=8)
            if r.ok: st.success("API is reachable")
            else: st.warning(f"API returned HTTP {r.status_code}")
        except requests.RequestException as e:
            st.error(f"Connection failed: {e}")
    st.divider()
    st.caption("Model: XGBoost")
    st.caption("Decision threshold: 0.536")

if page == "Overview":
    st.caption("WORKSPACE / OVERVIEW")
    st.title("Customer Churn Overview")
    st.write("Monitor churn risk and explore predictions from your machine learning model.")
    cols = st.columns(4)
    values = [("CUSTOMERS IN DATASET", "7,043", "IBM Telco dataset"),
              ("HISTORICAL CHURN RATE", "26.54%", "Dataset distribution"),
              ("MODEL", "XGBoost", "Trained classification pipeline"),
              ("TEST PR-AUC", "67.28%", "Held-out test result")]
    for c, v in zip(cols, values):
        with c: metric(*v)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Customer distribution")
        df = pd.DataFrame({"Status":["Stayed","Churned"],"Customers":[5174,1869]})
        fig = px.pie(df, names="Status", values="Customers", hole=.65,
                     color="Status", color_discrete_map={"Stayed":"#2F6B9A","Churned":"#E58B65"})
        fig.update_layout(margin=dict(t=10,b=10), legend_orientation="h")
        show_chart(fig)
    with right:
        st.subheader("Final test performance")
        df = pd.DataFrame({"Metric":["Recall","ROC-AUC","PR-AUC","Precision","F1-score"],
                           "Score":[78.21,85.51,67.28,53.16,63.29]})
        fig = px.bar(df.sort_values("Score"), x="Score", y="Metric", orientation="h",
                     text=df.sort_values("Score")["Score"].map(lambda x:f"{x:.2f}%"),
                     color_discrete_sequence=["#2F6B9A"])
        fig.update_layout(margin=dict(t=10,b=10), xaxis_title="Score (%)", yaxis_title="")
        fig.update_traces(textposition="outside", cliponaxis=False)
        show_chart(fig)
    st.info("These are recorded dataset and test metrics, not live counts from a connected customer database.")

elif page == "Churn Prediction":
    st.caption("WORKSPACE / PREDICTION")
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to estimate the probability of churn.")
    st.caption("All 19 original model features are collected below. Engineered features are generated by the API.")

    with st.form("prediction_form"):
        st.subheader("1. Personal information")
        a,b,c = st.columns(3)
        gender = a.selectbox("Gender", ["Female","Male"])
        senior = b.selectbox("Senior citizen", [0,1], format_func=lambda x:"Yes" if x else "No")
        partner = c.selectbox("Has partner?", ["Yes","No"])
        d,e = st.columns(2)
        dependents = d.selectbox("Has dependents?", ["Yes","No"])
        tenure = e.number_input("Tenure (months)", 0, 72, 12)

        st.divider()
        st.subheader("2. Phone and internet services")
        a,b,c = st.columns(3)
        phone = a.selectbox("Phone service", ["Yes","No"])
        lines = b.selectbox("Multiple lines", ["No","Yes","No phone service"])
        internet = c.selectbox("Internet service", ["DSL","Fiber optic","No"])
        internet_opts = ["Yes","No","No internet service"]
        a,b,c = st.columns(3)
        security = a.selectbox("Online security", internet_opts)
        backup = b.selectbox("Online backup", internet_opts)
        device = c.selectbox("Device protection", internet_opts)
        a,b,c = st.columns(3)
        support = a.selectbox("Tech support", internet_opts)
        tv = b.selectbox("Streaming TV", internet_opts)
        movies = c.selectbox("Streaming movies", internet_opts)

        st.divider()
        st.subheader("3. Contract and billing")
        a,b = st.columns(2)
        contract = a.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless = b.selectbox("Paperless billing", ["Yes","No"])
        payment = st.selectbox("Payment method", [
            "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
        ])

        st.divider()
        st.subheader("4. Charges")
        a,b = st.columns(2)
        monthly = a.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
        total = b.number_input("Total charges ($)", min_value=0.0, value=float(70*12), step=1.0)

        submit = st.form_submit_button("Predict customer churn", type="primary", use_container_width=True)

    if submit:
        internet_fields = [security,backup,device,support,tv,movies]
        if internet == "No" and any(x != "No internet service" for x in internet_fields):
            st.error("When Internet service is No, all internet-related services must be 'No internet service'.")
            st.stop()
        if internet != "No" and any(x == "No internet service" for x in internet_fields):
            st.error("Choose Yes or No for internet-related services when internet is active.")
            st.stop()
        if phone == "No" and lines != "No phone service":
            st.error("When Phone service is No, Multiple lines must be 'No phone service'.")
            st.stop()
        if phone == "Yes" and lines == "No phone service":
            st.error("When Phone service is Yes, choose Yes or No for Multiple lines.")
            st.stop()
        if tenure == 0 and total != 0:
            st.error("For tenure 0, set Total charges to 0.")
            st.stop()

        features = {
            "gender":gender, "SeniorCitizen":int(senior), "Partner":partner,
            "Dependents":dependents, "tenure":int(tenure), "PhoneService":phone,
            "MultipleLines":lines, "InternetService":internet,
            "OnlineSecurity":security, "OnlineBackup":backup,
            "DeviceProtection":device, "TechSupport":support,
            "StreamingTV":tv, "StreamingMovies":movies, "Contract":contract,
            "PaperlessBilling":paperless, "PaymentMethod":payment,
            "MonthlyCharges":float(monthly), "TotalCharges":float(total)
        }
        with st.spinner("Requesting prediction from FastAPI..."):
            try:
                response = requests.post(f"{api_url()}/predict", json={"features":features}, timeout=60)
                if response.ok:
                    st.session_state.prediction = {"response":response.json(), "features":features}
                    st.success("Prediction completed.")
                else:
                    try: detail = response.json()
                    except ValueError: detail = response.text
                    st.error(f"API error ({response.status_code}): {detail}")
            except requests.ConnectionError:
                st.error("Cannot connect to FastAPI. Start the backend and verify the API URL.")
            except requests.Timeout:
                st.error("API request timed out. Please try again.")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    if st.session_state.prediction:
        data = st.session_state.prediction["response"]
        pred, prob, threshold, model = normalize_result(data)
        st.divider()
        st.subheader("Prediction result")
        if prob is None:
            st.warning("API response did not contain a recognized churn probability field.")
            st.json(data)
        else:
            churn = str(pred).lower() == "churn" if pred is not None else prob >= threshold
            a,b,c = st.columns(3)
            with a: metric("PREDICTION", "CHURN" if churn else "NO CHURN", "Model classification")
            with b: metric("CHURN PROBABILITY", f"{prob:.2f}%", "Estimated probability")
            with c: metric("DECISION THRESHOLD", f"{threshold:.2f}%", model)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob, number={"suffix":"%"},
                title={"text":"Estimated churn probability"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#B45309" if churn else "#167A55"},
                       "steps":[{"range":[0,30],"color":"#E8F3EE"},
                                {"range":[30,60],"color":"#FFF3D6"},
                                {"range":[60,100],"color":"#FCE8E6"}],
                       "threshold":{"line":{"color":"#172033","width":3},"thickness":.8,"value":threshold}}
            ))
            fig.update_layout(height=280, margin=dict(t=55,b=10,l=20,r=20))
            show_chart(fig)
            if churn:
                st.warning("Higher churn risk. Review the customer's contract, service experience, and support needs.")
            else:
                st.success("Lower churn risk at the configured decision threshold.")
            with st.expander("View submitted customer features"):
                st.json(st.session_state.prediction["features"])

elif page == "Model Analytics":
    st.caption("WORKSPACE / MODEL ANALYTICS")
    st.title("Model Performance")
    st.write("Final held-out test metrics for the selected XGBoost model.")
    metrics = [("ACCURACY","75.97%","Test set"),("PRECISION","53.16%","Test set"),
               ("RECALL","78.21%","Test set"),("F1-SCORE","63.29%","Test set"),
               ("ROC-AUC","85.51%","Test set")]
    cols = st.columns(5)
    for c,v in zip(cols,metrics):
        with c: metric(*v)
    left,right = st.columns(2, gap="large")
    with left:
        st.subheader("Classification metrics")
        df = pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1-score","ROC-AUC","PR-AUC"],
                           "Score":[75.97,53.16,78.21,63.29,85.51,67.28]})
        fig = px.bar(df, x="Metric", y="Score", text=df["Score"].map(lambda x:f"{x:.2f}%"),
                     color_discrete_sequence=["#2F6B9A"])
        fig.update_layout(yaxis_range=[0,100], yaxis_title="Score (%)", xaxis_title="")
        fig.update_traces(textposition="outside", cliponaxis=False)
        show_chart(fig)
    with right:
        st.subheader("Confusion matrix")
        cm = pd.DataFrame([[584,193],[61,219]],
                          index=["Actual: No Churn","Actual: Churn"],
                          columns=["Predicted: No Churn","Predicted: Churn"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted",y="Actual",color="Customers"))
        show_chart(fig)
    st.subheader("Model configuration")
    st.dataframe(pd.DataFrame({
        "Configuration":["Algorithm","Pipeline","Decision threshold","Primary selection metric","Test samples"],
        "Value":["XGBoost Classifier","Preprocessing + XGBoost","0.536","Average Precision (PR-AUC)","1,057"]
    }), use_container_width=True, hide_index=True)
    st.caption("Recorded final test metrics. Do not use the held-out test set repeatedly for model tuning.")

st.divider()
st.caption("Customer Churn Prediction • Streamlit + FastAPI + XGBoost | Predictions are estimates to support business decisions.")
