import json
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="InsuraSense | Apex Insure",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "health_premium_model.pkl"
SCHEMA_PATH = BASE_DIR / "feature_schema.json"
BACKGROUND_PATH = BASE_DIR / "assets" / "background.jpg"


# ==================================================
# 3. BACKGROUND IMAGE
# ==================================================

def get_background_image():
    """Return the local background image as a Base64 data URL."""

    if not BACKGROUND_PATH.is_file():
        return None

    try:
        with open(BACKGROUND_PATH, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    except OSError:
        return None


background_image = get_background_image()


# ==================================================
# 4. CUSTOM CSS
# ==================================================

if background_image:
    background_style = f"""
        background-image:
            linear-gradient(
                rgba(7, 17, 31, 0.82),
                rgba(7, 17, 31, 0.92)
            ),
            url("{background_image}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: #07111f;
    """
else:
    background_style = """
        background:
            radial-gradient(
                circle at top right,
                #173b56 0%,
                #0b172a 48%,
                #07111f 100%
            );
    """


st.markdown(
    f"""
    
    <style>

    /* Main background */

    .stApp {{
        {background_style}
    }}

    /* Main content container */

    .block-container {{
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Hide unnecessary interface elements */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Hero header */

    .hero {{
        padding: 32px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(15, 35, 61, 0.97),
            rgba(14, 76, 94, 0.92)
        );
        border: 1px solid rgba(94, 234, 212, 0.30);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25);
        margin-bottom: 26px;
    }}

    .brand {{
        color: #5eead4;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .hero h1 {{
        color: #ffffff !important;
        font-size: 43px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1.2;
    }}

    .hero-description {{
        color: #d2e4ef !important;
        font-size: 15px;
        line-height: 1.8;
        margin-top: 15px;
        max-width: 620px;
    }}

    .hero-badge {{
        display: inline-block;
        margin-top: 18px;
        padding: 8px 13px;
        border-radius: 30px;
        color: #99f6e4 !important;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid rgba(94, 234, 212, 0.30);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    /* Headings */

    h2, h3 {{
        color: #f1f5f9 !important;
    }}

    /* Description below the input heading */

    .section-description {{
        color: #b6c8d9 !important;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 4px;
        margin-bottom: 20px;
    }}

    /* Input controls */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {{
        background-color: rgba(10, 23, 40, 0.95);
        border-color: #40566d;
        border-radius: 10px;
    }}

    div[data-baseweb="input"] input {{
        color: #ffffff !important;
    }}

    div[data-baseweb="select"] span {{
        color: #ffffff;
    }}

    /* Input labels */

    [data-testid="stWidgetLabel"] p {{
        color: #e2eaf3 !important;
        font-weight: 500;
    }}

    /* Predict button */

    .stButton > button {{
        width: 100%;
        min-height: 50px;
        border-radius: 12px;
        border: 1px solid transparent;
        background: linear-gradient(90deg, #0d9488, #14b8a6);
        color: #ffffff;
        font-size: 16px;
        font-weight: 700;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        background: linear-gradient(90deg, #0f766e, #0d9488);
        border-color: #5eead4;
        color: #ffffff;
    }}

    /* Prediction result */

    .result-card {{
        background: linear-gradient(
            135deg,
            rgba(13, 148, 136, 0.24),
            rgba(15, 35, 61, 0.97)
        );
        border: 1px solid rgba(94, 234, 212, 0.45);
        border-radius: 20px;
        padding: 28px 20px;
        margin-top: 22px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.18);
    }}

    .result-label {{
        color: #99f6e4 !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .result-amount {{
        color: #ffffff !important;
        font-size: 36px;
        font-weight: 800;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }}

    .result-note {{
        color: #c3d5e4 !important;
        font-size: 13px;
        margin-top: 12px;
        line-height: 1.7;
    }}

    /* Expanders */

    div[data-testid="stExpander"] {{
        background: rgba(15, 29, 48, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
    }}

    div[data-testid="stExpander"] summary p {{
        color: #e2eaf3 !important;
    }}

    /* Dataframe */

    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Footer */

    .footer {{
        text-align: center;
        color: #a5b8ca !important;
        font-size: 12px;
        padding-top: 20px;
        line-height: 1.9;
    }}

    .footer strong {{
        color: #5eead4;
    }}

    /* Mobile layout */

    @media (max-width: 600px) {{
        .hero {{
            padding: 24px 20px;
        }}

        .hero h1 {{
            font-size: 34px;
        }}

        .result-amount {{
            font-size: 29px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 5. HERO HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="brand">
            APEX INSURE · INTELLIGENT HEALTHCARE
        </div>
        <h1>InsuraSense</h1>
        <div class="hero-description">
            Make informed healthcare insurance decisions with
            machine-learning-powered premium estimation.
            Enter applicant details to receive an estimated
            insurance premium in seconds.
        </div>
        <div class="hero-badge">
            ✦ AI-POWERED PREMIUM ESTIMATION
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 6. LOAD MODEL AND SCHEMA
# ==================================================

@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.is_file() or not SCHEMA_PATH.is_file():
        st.error(
            "Model artifacts not found. "
            "Please check health_premium_model.pkl "
            "and feature_schema.json."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return model, payload


model, payload = load_artifacts()

schema = payload["schema"]
metrics = payload.get("metrics", {})


# ==================================================
# 7. MODEL INFORMATION
# ==================================================

with st.expander("📊 View model information"):

    st.write("**Prediction target:**", schema.get("target"))

    st.write(
        "**Features:**",
        [feature["name"] for feature in schema["features"]]
    )

    if metrics:
        st.write("**Model evaluation metrics:**")
        st.json(metrics)


# ==================================================
# 8. APPLICANT INPUTS
# ==================================================

st.markdown(
    f"""
    <div style="margin-top: 24px;">
        <h2>Applicant details</h2>
        <div class="section-description">
            Provide the applicant's information below to
            estimate their health insurance premium.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

inputs = {}

# Preserve the original dynamic input fields.

for feat in schema["features"]:

    name = feat["name"]

    if feat["type"] == "numeric":

        val = st.number_input(
            name,
            value=float(feat.get("median", 0.0)),
            key=f"input_{name}"
        )

    else:

        vals = feat.get("values", [])

        val = st.selectbox(
            name,
            vals if vals else [""],
            key=f"input_{name}"
        )

    inputs[name] = val


# ==================================================
# 9. PREDICT INSURANCE PREMIUM
# ==================================================

if st.button(
    "✨ Predict Insurance Premium",
    use_container_width=True
):

    X = pd.DataFrame([inputs])

    try:

        pred = model.predict(X)[0]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Estimated Insurance Premium
                </div>
                <div class="result-amount">
                    ₹{pred:,.2f}
                </div>
                <div class="result-note">
                    This is a machine-learning estimate,
                    not a guaranteed insurance quote.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("View submitted applicant data"):
            st.dataframe(X, use_container_width=True)

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <strong>InsuraSense</strong> · Powered by Apex Insure
        <br>
        Built with Streamlit, Pandas and Scikit-learn
        <br>
        Predictions are estimates for demonstration purposes
        and are not guaranteed insurance quotes.
    </div>
    """,
    unsafe_allow_html=True
)