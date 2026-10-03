# pip install streamlit requests gtts
# Run: streamlit run app_farmer.py   (keep crop_model.pkl, and ideally Crop_recommendation.csv, in the same folder)

import base64, glob, hashlib, io, os, re
from datetime import date, timedelta

import joblib, numpy as np, pandas as pd, requests, streamlit as st
from gtts import gTTS

st.set_page_config(page_title="Crop Advisor", page_icon="🌾", layout="centered", initial_sidebar_state="auto")
GREEN, ORANGE = "#1E9B47", "#FF8A00"
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------- Style (light, green + orange, photo heroes) ----------------
st.markdown("""<style>
.stApp {background:#f6f5f1;color:#12301d}
.block-container {max-width:720px;padding-top:.8rem;padding-bottom:8rem}
footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden}
header {background:transparent}
[data-testid="stSidebar"] {background:#fff;min-width:360px}
[data-testid="stSidebar"] .stButton>button {min-height:46px;border-radius:14px;padding:0}
[data-testid="stSidebar"] .stButton>button p {font-size:1.25rem}
h1,h2,h3,p,label,span {color:#12301d}
div.stButton {width:100%}
[data-testid="stFileUploader"] section {border-radius:24px;background:#fff;border:2px dashed __G__}
.stButton>button {width:100%;min-height:92px;border-radius:30px;border:2px solid #e3e1da;background:#fff;
  box-shadow:0 6px 18px rgba(0,0,0,.06)}
.stButton>button p {font-size:1.5rem;font-weight:800;color:#12301d}
.stButton>button:hover {border-color:__G__;background:#f0faf3}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"] {background:__G__;border:none}
.stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p {color:#fff}
.stTextInput input {font-size:1.5rem;height:3.4rem;border-radius:40px;background:#fff;color:#12301d;padding-left:22px}
.stRadio label p {font-size:1.35rem}
.hero {border-radius:38px;background-size:cover;background-position:center;padding:24px 26px;display:flex;
  flex-direction:column;justify-content:flex-end;box-shadow:0 12px 30px rgba(0,0,0,.18);margin-bottom:14px}
.hero h1,.hero h2,.hero p,.hero span {color:#fff;margin:0}
.hero h1 {font-size:2.4rem;line-height:1.15}
.hero .g {color:#7CFFA8}
.card {background:#fff;border-radius:28px;padding:16px 20px;margin:12px 0;font-size:1.45rem;box-shadow:0 6px 18px rgba(0,0,0,.06)}
.best {background:__O__} .best, .best b, .best span {color:#fff}
.best .bar {background:rgba(255,255,255,.35)} .best .bar>div {background:#fff}
.big {font-size:3rem;line-height:1}
.bar {height:14px;border-radius:7px;background:#e8e6df;margin-top:8px}
.bar>div {height:14px;border-radius:7px;background:__G__}
.chip {display:inline-block;background:#fff;border-radius:24px;padding:10px 18px;margin:4px 6px 4px 0;font-size:1.4rem;
  box-shadow:0 4px 12px rgba(0,0,0,.06)}
.chip.on {background:__O__;color:#fff}
[class*="st-key-listen"] .stButton>button {min-height:54px;background:#fff}
[class*="st-key-listen"] .stButton>button p {font-size:1.15rem;color:__G__}
.st-key-nav {position:fixed;bottom:12px;left:50%;transform:translateX(-50%);width:min(690px,94vw);
  background:#fff;border-radius:34px;padding:8px;z-index:99;box-shadow:0 8px 30px rgba(0,0,0,.2)}
.st-key-nav .stButton>button {min-height:58px;background:#f0faf3;border:none;border-radius:26px;box-shadow:none}
.st-key-nav .stButton>button p {font-size:1.3rem;color:__G__}
.stSelectbox div[data-baseweb="select"]>div {min-height:68px;border-radius:34px;background:#fff;font-size:1.5rem;padding-left:14px}
.stSelectbox div[data-baseweb="select"] * {color:#12301d}
</style>""".replace("__G__", GREEN).replace("__O__", ORANGE), unsafe_allow_html=True)

# ---------------- Photos (put your images in an "assets" folder next to this file) ----------------
@st.cache_data(show_spinner=False)
def photo(path_no_ext):
    for f in glob.glob(path_no_ext + ".*"):
        ext = f.rsplit(".", 1)[1].lower()
        if ext in ("jpg", "jpeg", "png", "webp"):
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64," + base64.b64encode(open(f, "rb").read()).decode()
    return None

def hero(name, height, html):
    src = photo(f"{BASE}/assets/{name}")
    bg = f"url({src})" if src else "linear-gradient(135deg,#2e9d4f,#0b3d1e)"  # green fallback if no photo yet
    st.markdown(f"<div class='hero' style='height:{height}px;background-image:linear-gradient(180deg,"
                f"rgba(0,0,0,.05) 30%,rgba(0,0,0,.65)),{bg}'>{html}</div>", unsafe_allow_html=True)

# ---------------- Text in 3 languages ----------------
T = {
 "en": dict(title="Crop Advisor", where="Where is your farm?", ph="Village or town", fetch="Get weather",
   wx_fail="Place not found. Try a nearby town.", next="Next", soil_q="What is your soil?",
   soil={"red": "Red soil", "black": "Black soil", "alluvial": "River soil", "laterite": "Laterite soil",
         "sandy": "Sandy soil", "unsure": "Not sure"},
   card="I have Soil Card", card_q="What does your soil card say?", lv=["Low", "Medium", "High"],
   nut=["Nitrogen (N)", "Phosphorus (P)", "Potash (K)"], phq="Soil pH", phv=["Acidic", "Normal", "Alkaline"],
   exact="I have exact numbers", show="Show crops", res="Best crops for you", best="Best crop",
   back="Back", home="Start", listen="Listen",
   est="This is only an estimate. Please test your soil at the nearest agriculture office."),
 "kn": dict(title="ಬೆಳೆ ಸಲಹೆ", where="ನಿಮ್ಮ ಹೊಲ ಎಲ್ಲಿದೆ?", ph="ಊರಿನ ಹೆಸರು", fetch="ಹವಾಮಾನ ತೋರಿಸಿ",
   wx_fail="ಊರು ಸಿಗಲಿಲ್ಲ. ಹತ್ತಿರದ ಪಟ್ಟಣ ಪ್ರಯತ್ನಿಸಿ.", next="ಮುಂದೆ", soil_q="ನಿಮ್ಮ ಮಣ್ಣು ಯಾವುದು?",
   soil={"red": "ಕೆಂಪು ಮಣ್ಣು", "black": "ಕಪ್ಪು ಮಣ್ಣು", "alluvial": "ಮೆಕ್ಕಲು ಮಣ್ಣು", "laterite": "ಜಂಬಿಟ್ಟಿಗೆ ಮಣ್ಣು",
         "sandy": "ಮರಳು ಮಣ್ಣು", "unsure": "ಗೊತ್ತಿಲ್ಲ"},
   card="ಮಣ್ಣು ಕಾರ್ಡ್ ಇದೆ", card_q="ನಿಮ್ಮ ಕಾರ್ಡ್‌ನಲ್ಲಿ ಏನಿದೆ?", lv=["ಕಡಿಮೆ", "ಮಧ್ಯಮ", "ಹೆಚ್ಚು"],
   nut=["ಸಾರಜನಕ (N)", "ರಂಜಕ (P)", "ಪೊಟ್ಯಾಶ್ (K)"], phq="ಮಣ್ಣಿನ pH", phv=["ಆಮ್ಲೀಯ", "ಸಾಮಾನ್ಯ", "ಕ್ಷಾರೀಯ"],
   exact="ನನ್ನ ಬಳಿ ನಿಖರ ಸಂಖ್ಯೆಗಳಿವೆ", show="ಬೆಳೆ ತೋರಿಸಿ", res="ನಿಮಗೆ ಸೂಕ್ತ ಬೆಳೆಗಳು", best="ಉತ್ತಮ ಬೆಳೆ",
   back="ಹಿಂದೆ", home="ಮೊದಲಿಗೆ", listen="ಕೇಳಿ",
   est="ಇದು ಅಂದಾಜು ಮಾತ್ರ. ಹತ್ತಿರದ ಕೃಷಿ ಕಚೇರಿಯಲ್ಲಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."),
 "hi": dict(title="फसल सलाह", where="आपका खेत कहाँ है?", ph="गाँव या शहर का नाम", fetch="मौसम देखें",
   wx_fail="जगह नहीं मिली। पास का शहर आज़माएँ।", next="आगे", soil_q="आपकी मिट्टी कौन सी है?",
   soil={"red": "लाल मिट्टी", "black": "काली मिट्टी", "alluvial": "जलोढ़ मिट्टी", "laterite": "लेटराइट मिट्टी",
         "sandy": "रेतीली मिट्टी", "unsure": "पता नहीं"},
   card="मिट्टी कार्ड है", card_q="आपके कार्ड में क्या लिखा है?", lv=["कम", "मध्यम", "ज़्यादा"],
   nut=["नाइट्रोजन (N)", "फॉस्फोरस (P)", "पोटाश (K)"], phq="मिट्टी का pH", phv=["अम्लीय", "सामान्य", "क्षारीय"],
   exact="मेरे पास सही संख्याएँ हैं", show="फसल दिखाएँ", res="आपके लिए सही फसलें", best="सबसे अच्छी फसल",
   back="पीछे", home="शुरू में", listen="सुनें",
   est="यह सिर्फ़ अंदाज़ा है। नज़दीकी कृषि कार्यालय में मिट्टी की जाँच कराएँ।"),
}
T["en"].update(where="Which district is your farm in?", town="Nearby town (optional, English)", refine="Update",
               tn_fail="Town not found. Showing district weather.")
T["kn"].update(where="ನಿಮ್ಮ ಹೊಲ ಯಾವ ಜಿಲ್ಲೆಯಲ್ಲಿದೆ?", town="ಹತ್ತಿರದ ಪಟ್ಟಣ (ಬೇಕಿದ್ದರೆ, English)", refine="ಬದಲಿಸಿ",
               tn_fail="ಪಟ್ಟಣ ಸಿಗಲಿಲ್ಲ. ಜಿಲ್ಲೆಯ ಹವಾಮಾನ ತೋರಿಸಲಾಗಿದೆ.")
T["hi"].update(where="आपका खेत किस जिले में है?", town="नज़दीकी कस्बा (ज़रूरी नहीं, English)", refine="अपडेट",
               tn_fail="कस्बा नहीं मिला। जिले का मौसम दिखाया गया है।")
T["en"]["or"] = "or"; T["kn"]["or"] = "ಅಥವಾ"; T["hi"]["or"] = "या"
T["en"].update(scan="Take a photo of your soil card", scan_ok="Read from your card. Please check below.",
               scan_fail="Could not read the card clearly. Please choose below.")
T["kn"].update(scan="ಮಣ್ಣು ಕಾರ್ಡ್‌ನ ಫೋಟೋ ತೆಗೆಯಿರಿ", scan_ok="ಕಾರ್ಡ್‌ನಿಂದ ಓದಲಾಗಿದೆ. ಕೆಳಗೆ ಸರಿಯಾಗಿದೆಯೇ ನೋಡಿ.",
               scan_fail="ಕಾರ್ಡ್ ಸ್ಪಷ್ಟವಾಗಿ ಓದಲಾಗಲಿಲ್ಲ. ಕೆಳಗೆ ಆರಿಸಿ.")
T["hi"].update(scan="मिट्टी कार्ड की फोटो लें", scan_ok="कार्ड से पढ़ा गया। नीचे जाँच लें।",
               scan_fail="कार्ड साफ़ नहीं पढ़ पाए। नीचे चुनें।")
CROP_EMOJI = {"apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫘", "coconut": "🥥", "coffee": "☕",
  "cotton": "☁️", "grapes": "🍇", "jute": "🌿", "kidneybeans": "🫘", "lentil": "🫘", "maize": "🌽", "mango": "🥭",
  "mothbeans": "🫘", "mungbean": "🌱", "muskmelon": "🍈", "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🫛",
  "pomegranate": "🔴", "rice": "🌾", "watermelon": "🍉"}
CROP_NAME = {
 "kn": dict(apple="ಸೇಬು", banana="ಬಾಳೆಹಣ್ಣು", blackgram="ಉದ್ದು", chickpea="ಕಡಲೆ", coconut="ತೆಂಗಿನಕಾಯಿ", coffee="ಕಾಫಿ",
   cotton="ಹತ್ತಿ", grapes="ದ್ರಾಕ್ಷಿ", jute="ಸೆಣಬು", kidneybeans="ರಾಜ್ಮಾ", lentil="ಮಸೂರ್ ಬೇಳೆ", maize="ಮೆಕ್ಕೆಜೋಳ",
   mango="ಮಾವು", mothbeans="ಮಟಕಿ", mungbean="ಹೆಸರುಕಾಳು", muskmelon="ಖರ್ಬೂಜ", orange="ಕಿತ್ತಳೆ", papaya="ಪಪ್ಪಾಯಿ",
   pigeonpeas="ತೊಗರಿ", pomegranate="ದಾಳಿಂಬೆ", rice="ಭತ್ತ", watermelon="ಕಲ್ಲಂಗಡಿ"),
 "hi": dict(apple="सेब", banana="केला", blackgram="उड़द", chickpea="चना", coconut="नारियल", coffee="कॉफी",
   cotton="कपास", grapes="अंगूर", jute="जूट", kidneybeans="राजमा", lentil="मसूर", maize="मक्का",
   mango="आम", mothbeans="मोठ", mungbean="मूंग", muskmelon="खरबूजा", orange="संतरा", papaya="पपीता",
   pigeonpeas="अरहर", pomegranate="अनार", rice="धान", watermelon="तरबूज"),
}
def crop_name(c, lang):
    return c.title() if lang == "en" else CROP_NAME[lang][c]

# ---------------- Model + soil helpers ----------------
@st.cache_resource
def load():
    return joblib.load("crop_model.pkl")
bundle = load(); model, le, FEATURES = bundle["model"], bundle["encoder"], bundle["features"]

FALLBACK_Q = {"N": (21, 37, 84), "P": (28, 51, 68), "K": (20, 32, 49)}
try:
    _df = pd.read_csv("Crop_recommendation.csv")
    SOIL_Q = {c: tuple(_df[c].quantile([0.25, 0.5, 0.75])) for c in ["N", "P", "K"]}
except Exception:
    SOIL_Q = FALLBACK_Q
PH_VALUES = [5.5, 6.8, 7.8]
# rough typical profiles (approximate, NOT lab values): (N, P, K level idx, pH idx)
SOIL_PROFILE = {"red": (0, 0, 0, 0), "black": (0, 1, 2, 2), "alluvial": (1, 1, 1, 1),
                "laterite": (0, 0, 0, 0), "sandy": (0, 0, 0, 1), "unsure": (1, 1, 1, 1)}
def vals(n, p, k, ph):
    return (float(SOIL_Q["N"][n]), float(SOIL_Q["P"][p]), float(SOIL_Q["K"][k]), PH_VALUES[ph])

# ---------------- Karnataka districts (HQ coordinates, approximate) ----------------
DIST = {  # English: (lat, lon, Kannada, Hindi)
 "Bagalkot": (16.18, 75.70, "ಬಾಗಲಕೋಟೆ", "बागलकोट"), "Ballari": (15.14, 76.92, "ಬಳ್ಳಾರಿ", "बल्लारी"),
 "Belagavi": (15.85, 74.50, "ಬೆಳಗಾವಿ", "बेलगावी"), "Bengaluru Rural": (13.20, 77.70, "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ", "बेंगलुरु ग्रामीण"),
 "Bengaluru Urban": (12.97, 77.59, "ಬೆಂಗಳೂರು ನಗರ", "बेंगलुरु शहरी"), "Bidar": (17.91, 77.52, "ಬೀದರ್", "बीदर"),
 "Chamarajanagar": (11.93, 76.94, "ಚಾಮರಾಜನಗರ", "चामराजनगर"), "Chikkaballapur": (13.43, 77.73, "ಚಿಕ್ಕಬಳ್ಳಾಪುರ", "चिक्काबल्लापुर"),
 "Chikkamagaluru": (13.32, 75.77, "ಚಿಕ್ಕಮಗಳೂರು", "चिकमगलूरु"), "Chitradurga": (14.23, 76.40, "ಚಿತ್ರದುರ್ಗ", "चित्रदुर्ग"),
 "Dakshina Kannada": (12.91, 74.86, "ದಕ್ಷಿಣ ಕನ್ನಡ", "दक्षिण कन्नड़"), "Davanagere": (14.46, 75.92, "ದಾವಣಗೆರೆ", "दावणगेरे"),
 "Dharwad": (15.46, 75.01, "ಧಾರವಾಡ", "धारवाड़"), "Gadag": (15.43, 75.63, "ಗದಗ", "गदग"),
 "Hassan": (13.00, 76.10, "ಹಾಸನ", "हासन"), "Haveri": (14.79, 75.40, "ಹಾವೇರಿ", "हावेरी"),
 "Kalaburagi": (17.33, 76.83, "ಕಲಬುರಗಿ", "कलबुर्गी"), "Kodagu": (12.42, 75.74, "ಕೊಡಗು", "कोडागु"),
 "Kolar": (13.14, 78.13, "ಕೋಲಾರ", "कोलार"), "Koppal": (15.35, 76.15, "ಕೊಪ್ಪಳ", "कोप्पल"),
 "Mandya": (12.52, 76.90, "ಮಂಡ್ಯ", "मांड्या"), "Mysuru": (12.30, 76.64, "ಮೈಸೂರು", "मैसूरु"),
 "Ramanagara": (12.72, 77.28, "ರಾಮನಗರ", "रामनगर"), "Raichur": (16.21, 77.36, "ರಾಯಚೂರು", "रायचूर"),
 "Shivamogga": (13.93, 75.57, "ಶಿವಮೊಗ್ಗ", "शिवमोग्गा"), "Tumakuru": (13.34, 77.10, "ತುಮಕೂರು", "तुमकुरु"),
 "Udupi": (13.34, 74.75, "ಉಡುಪಿ", "उडुपी"), "Uttara Kannada": (14.80, 74.13, "ಉತ್ತರ ಕನ್ನಡ", "उत्तर कन्नड़"),
 "Vijayanagara": (15.27, 76.39, "ವಿಜಯನಗರ", "विजयनगर"), "Vijayapura": (16.83, 75.71, "ವಿಜಯಪುರ", "विजयपुरा"),
 "Yadgir": (16.77, 77.14, "ಯಾದಗಿರಿ", "यादगीर"),
}
def dist_label(d):
    return d if S.lang == "en" else DIST[d][2 if S.lang == "kn" else 3]

# ---------------- Weather (Open-Meteo) ----------------
def fill_weather():
    """Weather for the picked district centre, or for an optional nearby town inside that district."""
    d = S.get("district")
    if not d:
        return
    lat, lon, label = DIST[d][0], DIST[d][1], d
    S.tnote = False
    try:
        town = (S.get("place") or "").strip()
        if town:
            res = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                               params={"name": town, "count": 10, "country_code": "IN"}, timeout=10).json().get("results") or []
            dist = lambda r: ((r["latitude"] - lat) ** 2 + (r["longitude"] - lon) ** 2) ** 0.5
            res = [r for r in res if r.get("admin1") == "Karnataka" and dist(r) < 1.2]
            if res:
                r0 = min(res, key=dist)
                lat, lon, label = r0["latitude"], r0["longitude"], f'{r0["name"]}, {d}'
            else:
                S.tnote = True  # town not found -> keep district centre
        end = date.today() - timedelta(days=5); start = end - timedelta(days=90)
        df = pd.DataFrame(requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto", "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"},
            timeout=20).json()["daily"])
        t_, h_ = df["temperature_2m_mean"].mean(), df["relative_humidity_2m_mean"].mean()
        r_ = df["precipitation_sum"].sum() / 3  # mm per month
        S.w = (min(max(t_, 8.0), 44.0), min(max(h_, 14.0), 100.0), min(max(r_, 20.0), 300.0))
        S.wplace = label; S.wmsg = "ok"
    except Exception:
        S.wmsg = "fail"

# ---------------- Voice ----------------
@st.cache_data(show_spinner=False)
def tts(text, lang):
    buf = io.BytesIO(); gTTS(text, lang=lang).write_to_fp(buf); return buf.getvalue()

def listen(text, key):
    with st.container(key=f"listen_{key}"):
        if st.button("🔊 " + t["listen"], key=f"btn_{key}"):
            try:
                st.audio(tts(text, S.lang), format="audio/mp3", autoplay=True)
            except Exception:
                st.warning("Audio needs internet.")

# ---------------- Soil card OCR (RapidOCR, light) ----------------
# ---- soil card parser ----
CARD_LIMITS = {"N": (280, 560), "P": (10, 25), "K": (110, 280)}  # typical Soil Health Card ratings, kg/ha (check your state chart)
def level_of(k, v):
    lo, hi = CARD_LIMITS[k]
    return 0 if v < lo else (1 if v <= hi else 2)
def ph_level(v):
    return 0 if v < 6.0 else (1 if v <= 7.5 else 2)

def parse_lines(lines):
    """lines: text rows from the card. Returns {"N"/"P"/"K"/"pH": (value or None, level 0/1/2)}."""
    found = {}
    for line in lines:
        low = line.lower().replace(",", ".")
        if re.search(r"\bp\s?\.?\s?h\b", low): key = "pH"
        elif "nitrogen" in low: key = "N"
        elif "phosph" in low or "p2o5" in low: key = "P"
        elif "potas" in low or "k2o" in low: key = "K"
        else: continue
        if key in found: continue
        clean = re.sub(r"p2o5|k2o|kg\s*/\s*ha|mg\s*/\s*kg|\(.*?\)", " ", low)
        clean = re.sub(r"^\s*\d+[.)]?\s+(?=[a-z])", "", clean)  # drop serial number at start of row
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", clean)]
        rating = re.search(r"\b(low|medium|high)\b", low)
        if key == "pH":
            nums = [n for n in nums if 3 <= n <= 10]
            if nums: found[key] = (nums[0], ph_level(nums[0]))
        elif rating:
            found[key] = (nums[0] if nums else None, {"low": 0, "medium": 1, "high": 2}[rating.group(1)])
        elif nums:
            found[key] = (nums[0], level_of(key, nums[0]))
    return found
# ---- end soil card parser ----

@st.cache_resource(show_spinner=False)
def get_engine():
    """Light CPU OCR (RapidOCR, ONNX). Falls back to EasyOCR if that is what is installed."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        eng = RapidOCR()
        return lambda arr: [(box, txt) for box, txt, _ in (eng(arr)[0] or [])]
    except ImportError:
        import easyocr
        rd = easyocr.Reader(["en"], gpu=False)
        return lambda arr: [(box, txt) for box, txt, _ in rd.readtext(arr)]

def scan_card(data):
    """OCR the photo, group words into table rows, parse N, P, K, pH."""
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width > 1400:  # smaller image = faster OCR on a weak CPU
        img = img.resize((1400, int(img.height * 1400 / img.width)))
    res = get_engine()(np.array(img))
    if not res:
        return {}
    tol = max(float(np.median([abs(b[2][1] - b[0][1]) for b, _ in res])) * 0.7, 8)
    items = sorted((((b[0][1] + b[2][1]) / 2, b[0][0], txt) for b, txt in res))
    rows, cur, y0 = [], [], None
    for y, x, txt in items:
        if y0 is None or abs(y - y0) <= tol:
            cur.append((x, txt)); y0 = y if y0 is None else y0
        else:
            rows.append(cur); cur = [(x, txt)]; y0 = y
    rows.append(cur)
    return parse_lines([" ".join(tx for _, tx in sorted(r)) for r in rows])

# ---------------- State + callbacks ----------------
S = st.session_state
S.setdefault("step", "lang"); S.setdefault("lang", "en"); S.setdefault("w", (25.0, 70.0, 100.0))
t = T[S.lang]

def go(step): S.step = step
def pick_from_select():
    if S.get("district_sel"):
        S.district = S.district_sel; S.place = ""; fill_weather()
def set_lang(code): S.lang = code; S.step = "where"
def pick_soil(key):
    if key == "card": S.step = "card"
    else: S.vals = vals(*SOIL_PROFILE[key]); S.est = True; S.step = "result"
def finish_card():
    S.vals = (S.ex_n, S.ex_p, S.ex_k, S.ex_ph) if S.get("use_exact") else vals(S.nv, S.pv, S.kv, S.phv) 
    S.est = False; S.step = "result"

def tiles(items, per_row, cb):
    """items: list of (key, label). Big tile buttons in a grid."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (k, label) in zip(cols, items[i:i + per_row]):
            col.button(label, key=f"tile_{k}", on_click=cb, args=(k,))

# ---------------- Screens ----------------
if S.step == "lang":
    hero("welcome", 470, "<h1>Smart Solutions<br><span class='g'>Modern Farmers</span></h1>"
                         "<p style='font-size:1.3rem;margin-top:8px'>ಬೆಳೆ ಸಲಹೆ · फसल सलाह · Crop Advisor</p>")
    for col, (code, name) in zip(st.columns(3), [("en", "English"), ("kn", "ಕನ್ನಡ"), ("hi", "हिन्दी")]):
        col.button(name, key=f"lang_{code}", on_click=set_lang, args=(code,))

elif S.step == "where":
    hero("field", 170, f"<h2>📍 {t['where']}</h2>"); listen(t["where"], "where")
    st.selectbox(t["where"], list(DIST), index=list(DIST).index(S.district) if S.get("district") else None,
                 format_func=dist_label, key="district_sel", on_change=pick_from_select,
                 label_visibility="collapsed", placeholder="▼  " + t["where"])
    if S.get("wmsg") == "fail":
        st.error(t["wx_fail"])
    elif S.get("wmsg") == "ok":
        tp, hu, rn = S.w
        st.markdown(f"<div><span class='chip on'>🌡️ {tp:.0f}°C</span><span class='chip'>💧 {hu:.0f}%</span>"
                    f"<span class='chip'>🌧️ {rn:.0f} mm</span><span class='chip'>📍 {S.wplace}</span></div>",
                    unsafe_allow_html=True)
        if S.get("tnote"):
            st.warning(t["tn_fail"])
    st.button("➡️ " + t["next"], key="to_soil", on_click=go, args=("soil",), type="primary",
              disabled=S.get("wmsg") != "ok")
    if S.get("district"):
        with st.expander("🏘️ " + t["town"]):
            st.text_input(t["town"], key="place", label_visibility="collapsed", placeholder=t["town"])
            st.button("🔄 " + t["refine"], key="refine", on_click=fill_weather)

elif S.step == "soil":
    hero("soil", 170, f"<h2>🌱 {t['soil_q']}</h2>"); listen(t["soil_q"], "soil")
    ico = {"red": "🔴", "black": "⚫", "alluvial": "🟤", "laterite": "🟠", "sandy": "🟡", "unsure": "❓"}
    st.button("📄 " + t["card"], key="tile_card", on_click=pick_soil, args=("card",), type="primary")
    st.markdown(f"<p style='text-align:center;font-size:1.3rem;margin:4px 0'>— {t['or']} —</p>", unsafe_allow_html=True)
    tiles([(k, f"{ico[k]} {v}") for k, v in t["soil"].items()], 2, pick_soil)

elif S.step == "card":
    hero("soil", 150, f"<h2>📄 {t['card_q']}</h2>"); listen(t["card_q"], "card")
    for k_ in ("nv", "pv", "kv", "phv"):
        S.setdefault(k_, 1)
    up = st.file_uploader(" " + t["scan"], type=["jpg", "jpeg", "png"], key="card_img")
    if up is not None:
        data = up.getvalue(); h = hashlib.md5(data).hexdigest()
        if S.get("ocr_hash") != h:  # new photo -> read it once, prefill the choices below
            S.ocr_hash = h
            try:
                with st.spinner("..."):
                    S.ocr_found = scan_card(data)
                for k_, sk in (("N", "nv"), ("P", "pv"), ("K", "kv"), ("pH", "phv")):
                    if k_ in S.ocr_found: S[sk] = S.ocr_found[k_][1]
            except ImportError:
                S.ocr_found = None; st.info("OCR needs:  pip install rapidocr-onnxruntime")
            except Exception:
                S.ocr_found = {}
        found = S.get("ocr_found")
        if found:
            st.success(t["scan_ok"])
            chips = "".join(f"<span class='chip'>{k_} {('%g' % v) + ' ' if v is not None else ''}→ "
                            f"{(t['phv'] if k_ == 'pH' else t['lv'])[lv]}</span>" for k_, (v, lv) in found.items())
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        elif found is not None:
            st.warning(t["scan_fail"])
    fmt = lambda i: ["⬇️", "➡️", "⬆️"][i] + " " + t["lv"][i]
    for key, name in zip(["nv", "pv", "kv"], t["nut"]):
        st.radio(name, [0, 1, 2], format_func=fmt, horizontal=True, key=key)
    st.radio(t["phq"], [0, 1, 2], format_func=lambda i: t["phv"][i], horizontal=True, key="phv")
    with st.expander("🔢 " + t["exact"]):
        st.checkbox("OK", key="use_exact")
        st.slider("N", 0, 140, 50, key="ex_n"); st.slider("P", 5, 145, 50, key="ex_p")
        st.slider("K", 5, 205, 50, key="ex_k"); st.slider("pH", 3.5, 9.9, 6.5, key="ex_ph")
    st.button("✅ " + t["show"], key="finish", on_click=finish_card, type="primary")

elif S.step == "result":
    N, P, K, ph = S.vals; tp, hu, rn = S.w
    probs = model.predict_proba(pd.DataFrame([[N, P, K, tp, hu, ph, rn]], columns=FEATURES))[0]
    top = [i for i in np.argsort(probs)[::-1][:3] if probs[i] >= 0.01]
    hero("result", 170, f"<h2>🏆 {t['res']}</h2>")
    speech = []
    for rank, i in enumerate(top):
        c = le.classes_[i]; name = crop_name(c, S.lang); pct = probs[i] * 100
        speech.append(f"{t['best']}: {name}." if rank == 0 else f"{name}.")
        img = photo(f"{BASE}/assets/crops/{c}")  # optional crop photo, falls back to emoji
        pic = (f"<img src='{img}' style='width:84px;height:84px;border-radius:22px;object-fit:cover'>" if img
               else f"<span class='big'>{CROP_EMOJI[c]}</span>")
        st.markdown(f"<div class='card {'best' if rank == 0 else ''}'><div style='display:flex;align-items:center;gap:16px'>"
                    f"{pic}<div style='flex:1'><b>{name}</b> &nbsp; {pct:.0f}%"
                    f"<div class='bar'><div style='width:{pct:.0f}%'></div></div></div></div></div>", unsafe_allow_html=True)
    if top:
        listen(" ".join(speech), "result")
    if S.get("est"):
        st.info(t["est"])

# ---------------- Bottom bar ----------------
if S.step != "lang":
    BACK = {"where": "lang", "soil": "where", "card": "soil", "result": "card" if not S.get("est", True) else "soil"}
    with st.container(key="nav"):
        b1, b2 = st.columns(2)
        b1.button("⬅️ " + t["back"], key="nav_back", on_click=go, args=(BACK[S.step],))
        b2.button("🏠 " + t["home"], key="nav_home", on_click=go, args=("lang",))