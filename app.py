import json
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title='Smart Health Insurance Predictor — Apex Insure', page_icon='💊', layout='centered')
st.title('InsuraSense')
st.caption('for **Apex Insure** — upload-free, on-device demo')

MODEL_PATH = Path('health_premium_model.pkl')
SCHEMA_PATH = Path('feature_schema.json')

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        st.error('Artifacts not found. Please run the training notebook or train_model.py first.')
        st.stop()
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return model, payload

model, payload = load_artifacts()
schema = payload['schema']
metrics = payload.get('metrics', {})

with st.expander('Model info'):
    st.write(metrics)
    st.write('Target:', schema.get('target'))
    st.write('Features:', [f['name'] for f in schema['features']])

st.subheader('Enter applicant details')
inputs = {}
for feat in schema['features']:
    name = feat['name']
    if feat['type'] == 'numeric':
        val = st.number_input(name, value=float(feat.get('median', 0.0)))
    else:
        vals = feat.get('values', [])
        val = st.selectbox(name, vals if vals else [''])
    inputs[name] = val

if st.button('Predict premium', use_container_width=True):
    X = pd.DataFrame([inputs])
    try:
        pred = model.predict(X)[0]
        st.success(f'Estimated premium: **₹{pred:,.2f}**')
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.stop()
    st.write('Raw inputs:', X)

st.markdown('---')
st.caption('Built with ❤️ using scikit-learn + Streamlit')
