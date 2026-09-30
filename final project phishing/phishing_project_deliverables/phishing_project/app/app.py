"""
Streamlit App - Phishing URL Behaviour Classification
Track 1 Capstone - Problem 02 - Learn Depth Academy LLP

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import re
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "..", "model")

st.set_page_config(page_title="Phishing URL Classifier", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 8%, rgba(0, 190, 170, 0.16), transparent 34%),
            radial-gradient(ellipse at 88% 82%, rgba(20, 132, 174, 0.18), transparent 38%),
            linear-gradient(145deg, #10191d 0%, #111f24 48%, #101719 100%);
        color: #eaf4f2;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background-image:
            linear-gradient(rgba(112, 198, 187, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(112, 198, 187, 0.045) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 88%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stDeployButton"] {
        display: none;
    }

    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load trained artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.title("🛡️ Phishing URL Behaviour Classifier")
st.caption("Track 1 Capstone · Cybersecurity · Learn Depth Academy LLP")
st.write(
    "Paste a URL below. The app automatically extracts the structural features "
    "the model was trained on (URL length, subdomains, special characters, "
    "SSL indicator, etc.) and predicts whether it looks **Legitimate** or **Phishing**."
)

url = st.text_input("Enter a URL to check", placeholder="https://example.com/login")

st.markdown("---")
with st.expander("⚙️ Advanced: manually adjust the harder-to-auto-detect signals"):
    st.caption(
        "Some original dataset features (SSL certificate state, domain age, web "
        "traffic rank, Google index, page rank) require live external lookups "
        "that this offline prototype does not call. Default = neutral (0) reflects "
        "the class-mode fallback in the trained model; override them here if you "
        "know the true values."
    )
    ssl_state = st.selectbox("SSL certificate state", ["Trusted (-1)", "Suspicious (0)", "Not trusted (1)"], index=1)
    domain_age = st.selectbox("Domain age", [">= 6 months (-1)", "< 6 months (1)"], index=0)
    web_traffic = st.selectbox("Website traffic rank", ["High/known (-1)", "Medium (0)", "Low/unranked (1)"], index=1)
    google_index = st.selectbox("Indexed by Google", ["Yes (-1)", "No (1)"], index=0)
    page_rank = st.selectbox("Page rank", ["Has rank (-1)", "No rank (1)"], index=1)

st.markdown("---")
predict_btn = st.button("🔍 Classify URL", type="primary", use_container_width=True)


# ------------------------------------------------------------------
# Feature extraction from a raw URL (mirrors the training features)
# ------------------------------------------------------------------
def extract_features(u: str) -> dict:
    if not re.match(r"^https?://", u):
        u = "http://" + u

    domain_match = re.findall(r"://([^/]+)/?", u)
    domain = domain_match[0] if domain_match else u

    f = {}

    # having_ip_address: 1 if domain is an IP address
    f["having_ip_address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0]) else -1

    # url_length
    n = len(u)
    f["url_length"] = -1 if n < 54 else (0 if n <= 75 else 1)

    # shortining_service
    shorteners = r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly"
    f["shortining_service"] = 1 if re.search(shorteners, u) else -1

    # having_at_symbol
    f["having_at_symbol"] = 1 if "@" in u else -1

    # double_slash_redirecting (// appearing after the protocol position)
    f["double_slash_redirecting"] = 1 if u.rfind("//") > 7 else -1

    # prefix_suffix (hyphen in domain)
    f["prefix_suffix"] = 1 if "-" in domain else -1

    # having_sub_domain (count of dots in domain)
    dots = domain.split(":")[0].count(".")
    f["having_sub_domain"] = -1 if dots <= 1 else (0 if dots == 2 else 1)

    # --- Features that need external/live lookups: use user-adjustable defaults ---
    f["sslfinal_state"] = int(ssl_state.split("(")[-1].strip(")"))
    f["domain_registration_length"] = int(domain_age.split("(")[-1].strip(")"))
    f["favicon"] = -1
    f["port"] = 1 if ":" in domain and not domain.endswith((":80", ":443")) else -1
    f["https_token"] = 1 if "https-" in domain.lower() or "-https" in domain.lower() else -1
    f["request_url"] = -1
    f["url_of_anchor"] = 0
    f["links_in_tags"] = 0
    f["sfh"] = -1
    f["submitting_to_email"] = 1 if "mailto:" in u.lower() else -1
    f["abnormal_url"] = -1
    f["redirect"] = 0
    f["on_mouseover"] = -1
    f["rightclick"] = -1
    f["popupwindow"] = -1
    f["iframe"] = -1
    f["age_of_domain"] = int(domain_age.split("(")[-1].strip(")"))
    f["dnsrecord"] = -1
    f["web_traffic"] = int(web_traffic.split("(")[-1].strip(")"))
    f["page_rank"] = int(page_rank.split("(")[-1].strip(")"))
    f["google_index"] = int(google_index.split("(")[-1].strip(")"))
    f["links_pointing_to_page"] = 0
    f["statistical_report"] = -1

    return f


def validate_url(u: str) -> bool:
    return bool(u) and bool(re.match(r"^(https?://)?[\w\-]+(\.[\w\-]+)+", u.strip()))


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
if predict_btn:
    if not validate_url(url):
        st.error("Please enter a valid, well-formed URL (e.g. https://example.com).")
    else:
        feats = extract_features(url.strip())
        # Build the row in the exact same column order the model was trained on
        row = np.array([[feats[c] for c in feature_names]])
        row_scaled = scaler.transform(row)

        pred = model.predict(row_scaled)[0]
        proba = model.predict_proba(row_scaled)[0]

        st.markdown("### Result")
        if pred == 1:
            st.error(f"🚩 **Likely PHISHING** — risk score: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ **Likely LEGITIMATE** — confidence: {proba[0]*100:.1f}%")

        with st.expander("See extracted features used for this prediction"):
            st.json(feats)

        st.caption(
            "⚠️ Educational prototype only — trained on the UCI Phishing Websites "
            "dataset with foundational ML methods. Not a substitute for a production "
            "security tool. Some features are approximated/defaulted since this app "
            "makes no live external calls (zero-cost, offline-friendly by design)."
        )

st.markdown("---")
st.caption("Model: trained on 11,055 labeled websites · 30 structural/lexical URL features · UCI Machine Learning Repository")
