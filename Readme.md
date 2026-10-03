# 🌾 AI-Based Crop Recommendation for Farmers

A machine-learning app that suggests the best crops for a farmer's land, using soil and weather details. It is built for Karnataka farmers, including older and non-literate users, with big buttons, voice, and Kannada / Hindi / English.

## Features

- **Crop model:** Random Forest / XGBoost trained on the Kaggle Crop Recommendation dataset (22 crops, ~99% test accuracy). It gives the **top 3 crops with match %**.
- **Simple app for farmers** (`app_farmer.py`):
  - Language choice: English, ಕನ್ನಡ, हिन्दी
  - **District dropdown** (all 31 Karnataka districts) that auto-fetches weather from Open-Meteo (last 90 days: temperature, humidity, rainfall per month). Optional "nearby town" for more local weather.
  - **Soil card first:** upload or photograph a Soil Health Card and the app reads N, P, K and pH with OCR. The values are pre-filled and the farmer can correct them.
  - No soil card? Pick the soil colour/type (red, black, river, laterite, sandy, not sure).
  - Exact N, P, K, pH numbers are also supported (for lab reports).
  - **Voice:** 🔊 reads each question and the result aloud (Kannada, Hindi or English).
  - Big tiles, emoji icons, photo banners, and a bottom Back / Start bar.
- **Explainability:** SHAP plots show why the model picks a crop.

## Project structure

```
crop-recommendation/
├── crop_model.py            # trains the model, makes plots, saves crop_model.pkl
├── app_farmer.py            # main farmer-friendly app (use this one)
├── app.py                   # older, simpler slider version (optional)
├── Crop_recommendation.csv  # download from Kaggle (see below)
├── crop_model.pkl           # created by crop_model.py
├── requirements.txt
├── README.md
├── REPORT.md                # project report: what is built, results, limits
└── assets/                  # your photos (optional, app works without them)
    ├── welcome.jpg          # first screen banner
    ├── field.jpg            # location screen banner
    ├── soil.jpg             # soil screens banner
    ├── result.jpg           # result screen banner
    └── crops/               # optional: rice.jpg, maize.jpg, ... (falls back to emoji)
```

## Setup

1. Python 3.10+ (tested with 3.12).
2. Install packages:
   ```
   pip install -r requirements.txt
   ```
3. Download **Crop_recommendation.csv** from Kaggle ("Crop Recommendation Dataset") and put it in the project folder.

## Run

**Step 1: train the model** (only once)
```
python crop_model.py
```
This prints the accuracy report and creates `crop_model.pkl`, `confusion_matrix.png`, `feature_importance.png`, `shap_summary.png` and `shap_rice.png`.

**Step 2: start the app**
```
streamlit run app_farmer.py
```

The first soil-card scan loads the OCR model, so it can take a few extra seconds.

## How the app works

1. Choose language.
2. Choose district, and the weather fills in automatically.
3. Scan the soil card, or pick the soil type.
4. Check the Low / Medium / High choices, then tap **Show crops**.
5. See the top 3 crops and listen to the answer.

## Notes and limits

- The Kaggle dataset is clean and partly synthetic, so ~99% accuracy is higher than real fields will give.
- Soil-card values (kg/ha) and the dataset's N, P, K units are not identical. Low / Medium / High is mapped to the dataset's 25th / 50th / 75th percentiles as an approximation.
- Weather is the **last 90 days**, not a forecast, on a grid of roughly 10–25 km.
- The OCR reads **English text and digits** only.
- Kannada and Hindi text should be checked by a native speaker.
- Voice (gTTS) and weather need an internet connection.
- This is a student project. Results are a guide, not a replacement for advice from the local agriculture office (KVK / Raitha Samparka Kendra).

See `REPORT.md` for details.

## Credits

- Dataset: Crop Recommendation Dataset (Kaggle, Atharva Ingle)
- Weather and geocoding: [Open-Meteo](https://open-meteo.com) (CC BY 4.0)
- OCR: RapidOCR (ONNX)
- Text to speech: gTTS
- Photos: add your photo credits here
