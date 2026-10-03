# pip install streamlit requests
# Run: streamlit run app.py   (keep crop_model.pkl in the same folder)

import streamlit as st
import joblib, numpy as np, pandas as pd, requests
from datetime import date, timedelta

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

@st.cache_resource
def load():
    return joblib.load("crop_model.pkl")

bundle = load()
model, le, FEATURES = bundle["model"], bundle["encoder"], bundle["features"]


# ---------- Soil helpers ----------
# Low/Medium/High -> 25th / 50th / 75th percentile of the training data.
# Uses Crop_recommendation.csv if present, else approximate fallback values.
FALLBACK_Q = {  # (low, medium, high)
    "N": (21, 37, 84), "P": (28, 51, 68), "K": (20, 32, 49), "ph": (5.97, 6.43, 6.92)}
try:
    _df = pd.read_csv("Crop_recommendation.csv")
    SOIL_Q = {c: tuple(_df[c].quantile([0.25, 0.5, 0.75]).round(2)) for c in ["N", "P", "K", "ph"]}
except Exception:
    SOIL_Q = FALLBACK_Q

LEVEL_IDX = {"Low": 0, "Medium": 1, "High": 2}
PH_VALUES = {"Acidic (below 6)": 5.5, "Neutral (6 to 7.5)": 6.8, "Alkaline (above 7.5)": 7.8}

# Rough typical soil-type profiles (approximate; NOT lab values)
SOIL_TYPES = {
    "Not sure": ("Medium", "Medium", "Medium", "Neutral (6 to 7.5)"),
    "Red soil": ("Low", "Low", "Low", "Acidic (below 6)"),
    "Black soil": ("Low", "Medium", "High", "Alkaline (above 7.5)"),
    "Alluvial soil": ("Medium", "Medium", "Medium", "Neutral (6 to 7.5)"),
    "Laterite soil": ("Low", "Low", "Low", "Acidic (below 6)"),
    "Sandy soil": ("Low", "Low", "Low", "Neutral (6 to 7.5)"),
}

def level_value(nutrient, level):
    return float(SOIL_Q[nutrient][LEVEL_IDX[level]])

# ---------- Open-Meteo helpers ----------
def geocode(place):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1, "country_code": "IN"}, timeout=10)
    res = r.json().get("results")
    if not res:
        return None
    return res[0]["latitude"], res[0]["longitude"], f'{res[0]["name"]}, {res[0].get("admin1", "")}'

def get_weather(lat, lon, days=90):
    # archive data lags ~5 days, so end 5 days ago
    end = date.today() - timedelta(days=5)
    start = end - timedelta(days=days)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto"}, timeout=20)
    d = pd.DataFrame(r.json()["daily"])
    temp = d["temperature_2m_mean"].mean()
    hum = d["relative_humidity_2m_mean"].mean()
    rain = d["precipitation_sum"].sum() / (days / 30)  # avg mm per month
    return temp, hum, rain

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def fill_weather():
    place = st.session_state.get("place", "").strip()
    if not place:
        st.session_state["msg"] = ("warning", "Type a place name first.")
        return
    try:
        g = geocode(place)
        if not g:
            st.session_state["msg"] = ("warning", "Place not found. Try a nearby town.")
            return
        lat, lon, name = g
        t, h, r = get_weather(lat, lon)
        st.session_state["temp"] = clamp(t, 8.0, 44.0)
        st.session_state["hum"] = clamp(h, 14.0, 100.0)
        st.session_state["rain"] = clamp(r, 20.0, 300.0)
        st.session_state["msg"] = ("success",
            f"Filled from {name}: {t:.1f}°C, {h:.0f}% humidity, {r:.0f} mm/month rain (last 90 days).")
    except Exception as e:
        st.session_state["msg"] = ("error", f"Could not fetch weather: {e}")

# ---------- UI ----------
st.session_state.setdefault("temp", 25.0)
st.session_state.setdefault("hum", 70.0)
st.session_state.setdefault("rain", 100.0)
st.title("🌾 AI Crop Recommendation")

st.subheader("1. Location (auto-fills weather)")
lc1, lc2 = st.columns([3, 1])
lc1.text_input("Village / town / district", key="place", placeholder="e.g. Mandya")
lc2.write("")
lc2.button("Fetch weather", on_click=fill_weather)
if "msg" in st.session_state:
    kind, text = st.session_state["msg"]
    getattr(st, kind)(text)

st.subheader("2. Soil details")
mode = st.radio("What soil information do you have?", [
    "I have exact values (soil test report)",
    "I have Low / Medium / High (Soil Health Card)",
    "I don't have soil data"])

estimate_only = False
if mode.startswith("I have exact"):
    s1, s2 = st.columns(2)
    N = s1.slider("Nitrogen (N)", 0, 140, 50)
    P = s1.slider("Phosphorus (P)", 5, 145, 50)
    K = s2.slider("Potassium (K)", 5, 205, 50)
    ph = s2.slider("Soil pH", 3.5, 9.9, 6.5)
elif mode.startswith("I have Low"):
    s1, s2 = st.columns(2)
    n_lv = s1.selectbox("Nitrogen (N)", list(LEVEL_IDX), index=1)
    p_lv = s1.selectbox("Phosphorus (P)", list(LEVEL_IDX), index=1)
    k_lv = s2.selectbox("Potassium (K)", list(LEVEL_IDX), index=1)
    ph_ch = s2.selectbox("Soil pH", list(PH_VALUES), index=1)
    N, P, K = level_value("N", n_lv), level_value("P", p_lv), level_value("K", k_lv)
    ph = PH_VALUES[ph_ch]
    st.caption(f"Converted to approx values: N={N:.0f}, P={P:.0f}, K={K:.0f}, pH={ph}")
else:
    soil = st.selectbox("Your soil type (look at the colour)", list(SOIL_TYPES))
    n_lv, p_lv, k_lv, ph_ch = SOIL_TYPES[soil]
    N, P, K = level_value("N", n_lv), level_value("P", p_lv), level_value("K", k_lv)
    ph = PH_VALUES[ph_ch]
    estimate_only = True
    st.warning("Estimate only. Get your soil tested at the nearest KVK or agriculture office for a better result.")

st.subheader("3. Weather")
w1, w2, w3 = st.columns(3)
temp = w1.slider("Temperature (°C)", 8.0, 44.0, key="temp")
hum = w2.slider("Humidity (%)", 14.0, 100.0, key="hum")
rain = w3.slider("Rainfall (mm/month)", 20.0, 300.0, key="rain")

if st.button("Recommend crops", type="primary"):
    row = pd.DataFrame([[N, P, K, temp, hum, ph, rain]], columns=FEATURES)
    probs = model.predict_proba(row)[0]
    top = np.argsort(probs)[::-1][:3]

    st.subheader("Top recommendations")
    shown = 0
    for i in top:
        p = probs[i] * 100
        if p < 1:
            continue
        st.write(f"**{le.classes_[i].title()}**: {p:.1f}% match")
        st.progress(min(int(p), 100))
        shown += 1
    if shown == 0:
        st.warning("No strong match found. Check your inputs.")
    if estimate_only:
        st.info("This result is based on estimated soil values, so treat it as a rough guide.")