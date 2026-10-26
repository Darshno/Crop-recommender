# pip install streamlit requests
# Run: streamlit run app.py   (keep crop_model.pkl in the same folder)

import base64
import streamlit as st
import joblib, numpy as np, pandas as pd, requests
from datetime import date, timedelta
from pathlib import Path
from evaluation.validation import validate_input_values
from evaluation.input_semantics import (
    build_prediction_input,
    record_rainfall_clamp,
    summarize_rainfall_response,
)
from regional_recommender import build_historical_evidence, load_regional_data, rank_with_regional_evidence

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

@st.cache_resource
def load():
    model_path = Path(__file__).resolve().parent / "crop_model.pkl"
    if not model_path.exists():
        st.error(f"Required model file is missing: {model_path.name}")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Could not load the model file {model_path.name}: {exc}")
        st.stop()

bundle = load()
model, le, FEATURES = bundle["model"], bundle["encoder"], bundle["features"]

@st.cache_resource
def load_regional():
    path = Path(__file__).resolve().parent / "data" / "regional" / "crop_yield.csv"
    try:
        return load_regional_data(path)
    except (OSError, ValueError) as exc:
        return None, str(exc)

_regional = load_regional()


# ---------- Soil helpers ----------
# Low/Medium/High -> 25th / 50th / 75th percentile of the training data.
# Uses Crop_recommendation.csv if present, else approximate fallback values.
FALLBACK_Q = {  # (low, medium, high)
    "N": (21, 37, 84), "P": (28, 51, 68), "K": (20, 32, 49), "ph": (5.97, 6.43, 6.92)}
try:
    _df = pd.read_csv(Path(__file__).resolve().parent / "Crop_recommendation.csv")
    SOIL_Q = {c: tuple(_df[c].quantile([0.25, 0.5, 0.75]).round(2)) for c in ["N", "P", "K", "ph"]}
except (OSError, ValueError, KeyError):
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
    temp, hum, rainfall = get_weather_with_provenance(lat, lon, days)
    return temp, hum, rainfall.transformed_value


def get_weather_with_provenance(lat, lon, days=90):
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
    divisor = days / 30
    rainfall = summarize_rainfall_response(
        d,
        source="Open-Meteo archive API",
        requested_start_date=start.isoformat(),
        requested_end_date=end.isoformat(),
        divisor=divisor,
        source_metadata={"endpoint": "https://archive-api.open-meteo.com/v1/archive"},
    )
    return temp, hum, rainfall

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
        t, h, rainfall = get_weather_with_provenance(lat, lon)
        r = rainfall.transformed_value
        st.session_state["temp"] = clamp(t, 8.0, 44.0)
        st.session_state["hum"] = clamp(h, 14.0, 100.0)
        st.session_state["rain"] = clamp(r, 20.0, 300.0)
        st.session_state["rainfall_provenance"] = record_rainfall_clamp(
            rainfall,
            final_value=st.session_state["rain"],
            lower=20.0,
            upper=300.0,
        )
        st.session_state["msg"] = ("success",
            f"Filled from {name}: {t:.1f}°C, {h:.0f}% humidity, {r:.0f} mm/month rain (last 90 days).")
    except Exception as e:
        st.session_state["msg"] = ("error", f"Could not fetch weather: {e}")

# ---------- White farmer-friendly workflow ----------
st.markdown("""
<style>
.stApp { background:#f6f7f1; color:#173d2b; }
.block-container { max-width:1100px; padding:2rem 1.25rem 4rem; }
h1,h2,h3 { color:#173d2b; letter-spacing:-.02em; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:1.4rem; }
.leaf { width:42px; height:42px; border-radius:14px 14px 14px 4px; background:#238a4b; color:white;
        display:flex; align-items:center; justify-content:center; font-size:1.35rem; }
.brand-title { font-size:1.35rem; font-weight:800; line-height:1.05; }
.brand-subtitle,.muted { color:#66766d; }
.brand-subtitle { font-size:.78rem; margin-top:3px; }
.hero { border-radius:28px; overflow:hidden; margin-bottom:1.5rem; }
.hero-copy { background:linear-gradient(90deg,rgba(13,52,34,.88),rgba(13,52,34,.35)); padding:3rem; color:white; }
.hero-copy h1,.hero-copy p { color:white; max-width:620px; }
.hero-copy h1 { font-size:clamp(2rem,4vw,3.35rem); margin:0 0 .65rem; }
.hero-copy p { font-size:1.05rem; margin:0; }
.stepbar { display:flex; gap:8px; margin:0 0 1.4rem; }
.step { flex:1; padding:.75rem 1rem; border-radius:14px; background:#e9eee9; color:#66766d;
        font-weight:700; font-size:.9rem; }
.step.active { background:#238a4b; color:white; }
.step.done { background:#eaf5ec; color:#238a4b; }
.card { background:white; border:1px solid #e3e9e2; border-radius:22px; padding:1.3rem 1.4rem;
        box-shadow:0 8px 24px rgba(27,65,43,.06); margin-bottom:1rem; }
.card-title { font-size:1.25rem; font-weight:800; color:#173d2b; margin-bottom:.2rem; }
.weather-card { background:#eaf5ec; border:1px solid #d5e9d8; border-radius:18px; padding:1rem; }
.metric { font-size:1.3rem; font-weight:800; color:#173d2b; }
.metric-label { font-size:.78rem; color:#66766d; }
.result-card { background:white; border:1px solid #e3e9e2; border-radius:22px; padding:1.25rem; margin:.7rem 0; }
.result-card.top { border:2px solid #238a4b; background:#eaf5ec; }
div.stButton > button { border-radius:12px; min-height:2.7rem; font-weight:700; }
div.stButton > button[kind="primary"] { background:#238a4b; border-color:#238a4b; }
div[data-testid="stProgressBar"] > div > div { background:#238a4b; }
@media (max-width:700px) { .hero-copy { padding:2rem 1.4rem; } .step { padding:.65rem .5rem; font-size:.75rem; } }
</style>
""", unsafe_allow_html=True)

def _set_step(step):
    st.session_state["workflow_step"] = step

def _start_over():
    for key in ("soil_values", "prediction", "msg", "rainfall_provenance"):
        st.session_state.pop(key, None)
    st.session_state["workflow_step"] = 1

def _render_steps(current):
    labels = ["1  Location & weather", "2  Soil details", "3  Recommendations"]
    html = '<div class="stepbar">'
    for index, label in enumerate(labels, 1):
        state = "active" if index == current else "done" if index < current else ""
        html += f'<div class="step {state}">{label}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

st.session_state.setdefault("workflow_step", 1)
st.session_state.setdefault("temp", 25.0)
st.session_state.setdefault("hum", 70.0)
st.session_state.setdefault("rain", 100.0)
st.markdown('<div class="brand"><div class="leaf">⌁</div><div><div class="brand-title">AgriSense</div><div class="brand-subtitle">Better Crops · Stronger Farmers</div></div></div>', unsafe_allow_html=True)
hero_path = Path(__file__).resolve().parent / "assets" / "field.jpg"
if hero_path.exists():
    encoded_hero = base64.b64encode(hero_path.read_bytes()).decode("ascii")
    st.markdown(f'<div class="hero" style="background-image:linear-gradient(90deg,rgba(13,52,34,.88),rgba(13,52,34,.35)),url(data:image/jpeg;base64,{encoded_hero});background-size:cover;background-position:center"><div class="hero-copy"><h1>Smart solutions for modern farmers</h1><p>A simple crop advisor to bring together your location, weather, and soil details.</p></div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="hero-copy"><h1>Smart solutions for modern farmers</h1><p>A simple crop advisor to bring together your location, weather, and soil details.</p></div>', unsafe_allow_html=True)

step = st.session_state["workflow_step"]
_render_steps(step)

if step == 1:
    st.markdown('<div class="card"><div class="card-title">Where is your farm?</div><div class="muted">Enter a nearby village, town, or district to fetch recent weather.</div></div>', unsafe_allow_html=True)
    st.text_input("Location", key="place", placeholder="e.g. Mandya", label_visibility="collapsed")
    if st.button("Fetch weather", type="primary", use_container_width=True):
        with st.spinner("Finding your location and recent weather..."):
            fill_weather()
    if "msg" in st.session_state:
        kind, text = st.session_state["msg"]
        getattr(st, kind)(text)
    if st.session_state.get("msg", (None, ""))[0] == "success":
        st.markdown(f"""<div class="weather-card"><div class="muted">Recent weather for {st.session_state.get('place', 'your location')}</div>
        <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:.7rem">
        <div><div class="metric">{st.session_state['temp']:.1f}°C</div><div class="metric-label">Temperature</div></div>
        <div><div class="metric">{st.session_state['hum']:.0f}%</div><div class="metric-label">Humidity</div></div>
        <div><div class="metric">{st.session_state['rain']:.0f} mm</div><div class="metric-label">Rainfall estimate</div></div>
        </div></div>""", unsafe_allow_html=True)
    st.caption("Weather is optional. You can continue with the displayed values and adjust them later.")
    if st.button("Continue to soil details", type="primary", use_container_width=True):
        _set_step(2)
        st.rerun()

elif step == 2:
    st.markdown('<div class="card"><div class="card-title">Tell us about your soil</div><div class="muted">Choose the information you have. A soil test gives the most direct numeric input.</div></div>', unsafe_allow_html=True)
    mode = st.radio("Soil information", ["I have exact values (soil test report)", "I have Low / Medium / High (Soil Health Card)", "I don't have soil data"], key="soil_mode", label_visibility="collapsed")
    estimate_only = False
    if mode.startswith("I have exact"):
        c1, c2 = st.columns(2)
        N = c1.slider("Nitrogen (N)", 0, 140, 50, key="exact_n")
        P = c1.slider("Phosphorus (P)", 5, 145, 50, key="exact_p")
        K = c2.slider("Potassium (K)", 5, 205, 50, key="exact_k")
        ph = c2.slider("Soil pH", 3.5, 9.9, 6.5, key="exact_ph")
    elif mode.startswith("I have Low"):
        c1, c2 = st.columns(2)
        n_lv = c1.selectbox("Nitrogen (N)", list(LEVEL_IDX), index=1, key="level_n")
        p_lv = c1.selectbox("Phosphorus (P)", list(LEVEL_IDX), index=1, key="level_p")
        k_lv = c2.selectbox("Potassium (K)", list(LEVEL_IDX), index=1, key="level_k")
        ph_ch = c2.selectbox("Soil pH", list(PH_VALUES), index=1, key="level_ph")
        N, P, K = level_value("N", n_lv), level_value("P", p_lv), level_value("K", k_lv)
        ph = PH_VALUES[ph_ch]
        st.info(f"These levels use approximate values from the model's training distribution: N={N:.0f}, P={P:.0f}, K={K:.0f}, pH={ph}.")
    else:
        soil = st.selectbox("Soil type", list(SOIL_TYPES), key="soil_type")
        n_lv, p_lv, k_lv, ph_ch = SOIL_TYPES[soil]
        N, P, K = level_value("N", n_lv), level_value("P", p_lv), level_value("K", k_lv)
        ph = PH_VALUES[ph_ch]
        estimate_only = True
        st.warning("This is an estimate based on a general soil type. For more dependable input, use a soil test.")
    st.markdown('<div class="card"><div class="card-title">Weather details</div><div class="muted">Review fetched values or adjust them if you have better information.</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Temperature (°C)", 8.0, 44.0, key="temp")
    hum = c2.slider("Humidity (%)", 14.0, 100.0, key="hum")
    rain = c3.slider("Rainfall (mm/month)", 20.0, 300.0, key="rain")
    st.session_state["soil_values"] = (N, P, K, ph, estimate_only)
    b1, b2 = st.columns(2)
    if b1.button("Back to location", use_container_width=True):
        _set_step(1)
        st.rerun()
    if b2.button("See recommendations", type="primary", use_container_width=True):
        _set_step(3)
        st.rerun()

else:
    st.markdown('<div class="card"><div class="card-title">Your crop recommendations</div><div class="muted">Review your inputs, then calculate the model ranking.</div></div>', unsafe_allow_html=True)
    soil_values = st.session_state.get("soil_values")
    if not soil_values:
        st.warning("Please complete the soil details first.")
        if st.button("Back to soil details", use_container_width=True):
            _set_step(2)
            st.rerun()
    else:
        N, P, K, ph, estimate_only = soil_values
        temp, hum, rain = st.session_state["temp"], st.session_state["hum"], st.session_state["rain"]
        st.markdown(f'<div class="weather-card">Inputs ready: N {N:.0f} · P {P:.0f} · K {K:.0f} · pH {ph:.1f} · {temp:.1f}°C · {hum:.0f}% humidity · {rain:.0f} mm rainfall</div>', unsafe_allow_html=True)
        if st.button("Calculate recommendations", type="primary", use_container_width=True):
            with st.spinner("Checking your inputs and ranking crops..."):
                input_values = dict(zip(FEATURES, [N, P, K, temp, hum, ph, rain]))
                semantics = build_prediction_input(input_values, training_ranges=bundle.get("training_ranges"),
                    field_metadata={"rainfall": {"origin": "weather_api", "transformations": ("weather_api_to_model_input",)}} if st.session_state.get("rainfall_provenance") else None,
                    rainfall=st.session_state.get("rainfall_provenance"))
                checked = validate_input_values(semantics.values, bundle.get("training_ranges"))
                if not checked["valid"]:
                    st.session_state["prediction"] = ("error", checked["errors"])
                else:
                    row = pd.DataFrame([[checked["values"][f] for f in FEATURES]], columns=FEATURES)
                    st.session_state["prediction"] = ("ok", model.predict_proba(row)[0], checked["warnings"])
        prediction = st.session_state.get("prediction")
        if prediction:
            if prediction[0] == "error":
                for issue in prediction[1]:
                    st.error(issue)
            else:
                probs, warnings = prediction[1], prediction[2]
                for warning in warnings:
                    st.warning(warning)
                st.caption("Model scores rank the choices. They are not probabilities of crop success, expected yield, or proof of suitability.")
                for rank, i in enumerate(np.argsort(probs)[::-1][:3]):
                    pct = probs[i] * 100
                    card_class = "result-card top" if rank == 0 else "result-card"
                    st.markdown(f'<div class="{card_class}"><div style="display:flex;justify-content:space-between"><strong>{"Top recommendation · " if rank == 0 else ""}{le.classes_[i].title()}</strong><strong>{pct:.1f}%</strong></div></div>', unsafe_allow_html=True)
                    st.progress(min(int(pct), 100))
                if estimate_only:
                    st.info("These results use estimated soil values and should be treated as a rough guide.")
                st.markdown('<div class="card"><div class="card-title">Optional regional context</div><div class="muted">Historical evidence is shown separately from the model ranking.</div></div>', unsafe_allow_html=True)
                rc1, rc2, rc3 = st.columns(3)
                regional_state = rc1.text_input("State", placeholder="e.g. Karnataka", key="regional_state")
                regional_season = rc2.selectbox("Season", ["Kharif", "Rabi", "Whole Year", "Summer", "Other"], key="regional_season")
                regional_year = rc3.number_input("Evaluation year", min_value=1998, max_value=2100, value=2021, step=1, key="regional_year")
                if regional_state:
                    if isinstance(_regional, tuple):
                        st.warning(f"Regional evidence unavailable: {_regional[1]}")
                    else:
                        evidence = build_historical_evidence(_regional.data, regional_state, regional_season, int(regional_year))
                        regional_result = rank_with_regional_evidence(probs, [str(c) for c in le.classes_], evidence, state=regional_state, season=regional_season)
                        if regional_result["fallback_used"]:
                            st.info(regional_result["reason"] + " The model ranking is unchanged.")
                        else:
                            for item in regional_result["hybrid_ranking"][:3]:
                                st.write(f"**{item['crop'].title()}** · historical evidence score {item['regional_score']:.3f}")
                            st.caption("Historical association is not a forecast or proof of suitability.")
        b1, b2 = st.columns(2)
        if b1.button("Back to soil details", use_container_width=True):
            _set_step(2)
            st.rerun()
        if b2.button("Start over", use_container_width=True):
            _start_over()
            st.rerun()

# The legacy widgets remain below for rollback reference but are unreachable.
st.stop()
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

st.subheader("4. Optional regional context")
rc1, rc2, rc3 = st.columns(3)
regional_state = rc1.text_input("State (historical evidence)", placeholder="e.g. Karnataka")
regional_season = rc2.selectbox("Season", ["Kharif", "Rabi", "Whole Year", "Summer", "Other"])
regional_year = rc3.number_input("Evaluation year", min_value=1998, max_value=2100, value=2021, step=1)
st.caption("Historical cultivation/yield is an association, not proof of suitability or causal benefit. It does not replace agronomic advice.")

if st.button("Recommend crops", type="primary"):
    values = dict(zip(FEATURES, [N, P, K, temp, hum, ph, rain]))
    semantics = build_prediction_input(
        values,
        training_ranges=bundle.get("training_ranges"),
        field_metadata={"rainfall": {
            "origin": "weather_api",
            "transformations": ("weather_api_to_model_input",),
        }} if st.session_state.get("rainfall_provenance") else None,
        rainfall=st.session_state.get("rainfall_provenance"),
    )
    checked = validate_input_values(semantics.values, bundle.get("training_ranges"))
    if not checked["valid"]:
        for issue in checked["errors"]:
            st.error(issue)
    else:
        for warning in checked["warnings"]:
            st.warning(warning)
        row = pd.DataFrame([[checked["values"][f] for f in FEATURES]], columns=FEATURES)
        probs = model.predict_proba(row)[0]
        top = np.argsort(probs)[::-1][:3]

        st.subheader("Top recommendations")
        st.caption("Scores rank the model's choices. They are not probabilities of crop success or expected yield.")
        shown = 0
        for i in top:
            p = probs[i] * 100
            if p < 1:
                continue
            st.write(f"**{le.classes_[i].title()}**: {p:.1f}% model score")
            st.progress(min(int(p), 100))
            shown += 1
        if shown == 0:
            st.warning("No strong match found. Check your inputs or seek local agricultural advice.")
        if estimate_only:
            st.info("This result is based on estimated soil values, so treat it as a rough guide.")
        if regional_state:
            if isinstance(_regional, tuple) and len(_regional) == 2 and isinstance(_regional[1], str):
                st.warning(f"Regional evidence unavailable: {_regional[1]}")
            else:
                evidence = build_historical_evidence(
                    _regional.data, regional_state, regional_season, int(regional_year)
                )
                regional_result = rank_with_regional_evidence(
                    probs, [str(c) for c in le.classes_], evidence,
                    state=regional_state, season=regional_season,
                )
                st.subheader("Historical regional evidence")
                st.caption("Separate from the soil-model prediction. The hybrid ranking uses a conservative 20% regional weight only when evidence is available.")
                if regional_result["fallback_used"]:
                    st.info(regional_result["reason"] + " Showing the unchanged soil-model ranking.")
                else:
                    for item in regional_result["hybrid_ranking"][:3]:
                        st.write(f"**{item['crop'].title()}**: hybrid score {item['hybrid_score']:.3f} "
                                 f"(regional evidence {item['regional_score']:.3f})")
                    if regional_result["regional_only_candidates"]:
                        names = ", ".join(item["Crop"] for item in regional_result["regional_only_candidates"][:5])
                        st.info(f"Historical candidates not predicted by the 22-class soil model: {names}.")
