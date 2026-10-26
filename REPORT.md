# Project Report: AI-Based Crop Recommendation for Farmers

Student: Darshan V (Darshno), 2nd year AI/ML Engineering, New Horizon College of Engineering, Bangalore

---

## 1. Project synopsis

**Introduction:** Farmers often find it hard to decide which crop to grow. The right crop depends on soil nutrients, rain, temperature, and more. This project uses Artificial Intelligence (AI) to look at these things and suggest the best crops for a farmer's land. The farmer gets the answer in a simple app.

**Need / scope:**
- Many farmers pick crops by habit or guesswork. A wrong choice can cause low yield and money loss.
- Soil testing reports are hard to understand for many people.
- Weather is becoming less predictable, so old methods may not work well.
- The project covers Karnataka first and can grow to other states later.

**Methodology:** collect soil, weather and crop data, clean it, train machine-learning models (Random Forest, XGBoost), explain results with SHAP, add live weather, and build a simple multi-language app.

**Expected outcomes:** a trained model that suggests the best crops with high accuracy, a working app showing the top 3 crops, charts that explain the decisions, and this report.

**Societal benefits:** helps farmers choose the right crop and earn more, saves water, fertilizer and money, reduces the risk of crop failure, and local-language and voice support helps farmers who are not comfortable with English or reading.

---

## 2. How the project idea was chosen

Three ideas were compared:

| Idea | Verdict |
|---|---|
| E-tongue for Dravya (Ayurvedic substance) identification | Most unique, but needs sensor hardware. A photo-based version was discussed (plain image classification, colorimetric e-tongue, hybrid) |
| Photo-based Dravya identification | Easier, but captures looks, not taste |
| **AI-based crop recommendation** | **Chosen:** easiest to finish, no hardware, real social benefit |

---

## 3. Machine-learning model (`crop_model.py`)

- **Dataset:** Kaggle Crop Recommendation Dataset, 2200 rows, 22 crops (20 samples per crop in the test split), no missing values.
- **Inputs (7):** N, P, K, temperature, humidity, pH, rainfall.
- **Method:** stratified 80/20 split, 5-fold cross-validation, Random Forest vs XGBoost, best model saved with `joblib`.

**Results (run on the student's machine)**

| Model | CV accuracy | Test accuracy |
|---|---|---|
| **Random Forest** | 0.9949 | **0.9932** |
| XGBoost | 0.9909 | 0.9864 |

- Random Forest was selected. Almost every crop scored precision/recall of 1.00. The few errors (recall 0.95) were blackgram, lentil and rice, with jute, maize and mothbeans showing precision 0.95.
- Sample prediction: `recommend(90, 42, 43, 20.8, 82, 6.5, 202)` returned rice 94.7%, jute 5.3%.
- **Outputs created:** `crop_model.pkl`, `confusion_matrix.png`, `feature_importance.png`, `shap_summary.png` (all crops), `shap_rice.png` (per crop).
- **Bug fixed:** newer SHAP versions return one 3D array for multi-class models, which crashed `summary_plot`. The plotting code was changed to use per-class slices.
- `recommend()` returns the **top 3 crops with probabilities**.

---

## 4. The farmer app (`app_farmer.py`, built with Streamlit)

### Screens
1. **Language:** English, ಕನ್ನಡ, हिन्दी in one row.
2. **District:** dropdown with all 31 Karnataka districts. Choosing one fetches weather instantly. Optional "nearby town" box for more local weather.
3. **Soil:** the **soil card option is first**. Below it are soil-type tiles (red, black, river, laterite, sandy, not sure).
4. **Soil card:** upload or photograph the card, and OCR pre-fills the Low / Medium / High choices. An expander allows exact N, P, K, pH numbers.
5. **Result:** top 3 crops with emoji or photo, local-language name, match % bar, best crop highlighted, 🔊 button to hear the answer.

### Key features
- **Live weather (Open-Meteo, free):** average temperature, humidity, and rainfall converted to mm per month over the last 90 days. Values are clamped to the dataset's range.
- **Soil card OCR (RapidOCR):** reads the table rows, finds pH, Nitrogen, Phosphorus, Potassium, and uses the printed Low / Medium / High rating, or converts numbers using typical Soil Health Card cut-offs (N 280/560, P 10/25, K 110/280 kg/ha). The parser was tested on sample rows and on a generated fake card image (all four values read correctly in about 2 seconds on CPU).
- **Low / Medium / High to numbers:** mapped to the 25th / 50th / 75th percentile of the training data.
- **No soil data:** typical profile per soil type (approximate), with a "get your soil tested" warning.
- **Voice:** Google Text-to-Speech reads questions and results in the chosen language.
- **Languages:** full UI text and crop names in English, Kannada, Hindi.
- **Design for older and non-literate users:** one question per screen, large tiles, emoji and colour cues, voice, minimal typing, consistent bottom bar. Theme follows a light green and orange reference design, with photo banners from an `assets` folder (green gradient if no photos are added).

### Problems found and fixed during development
| Problem | Fix |
|---|---|
| SHAP multi-class `IndexError` | Use per-class SHAP slices |
| Streamlit session-state warning on weather sliders | Set defaults with `setdefault`, remove default values from sliders |
| Language buttons small and stacked | Placed in one row of equal columns, buttons stretch to full width |
| Keyboard idea (Kannada / Hindi on-screen) | Replaced by a district dropdown, which is simpler for farmers |
| EasyOCR too heavy (PyTorch, slow on CPU, hard to host) | Switched to RapidOCR (ONNX), fallback to EasyOCR if installed |

---

## 5. Project files

| File | Purpose |
|---|---|
| `crop_model.py` | Train, compare, evaluate, explain, save the model |
| `app_farmer.py` | Main farmer-friendly app |
| `app.py` | Older simple slider app (kept for reference) |
| `requirements.txt` | Python packages |
| `README.md` | Setup and run guide |
| `REPORT.md` | This report |
| `assets/` | Photos for banners and crops (to be added) |

---

## 6. Limitations (to state honestly)

- The Kaggle dataset is clean and partly synthetic, so ~99% accuracy is **inflated**. Real-world accuracy will be lower.
- Soil-card units (kg/ha) and the dataset's N, P, K units are not the same, so the Low / Medium / High mapping is an approximation.
- Soil-type profiles are rough general knowledge, not lab data.
- Weather is the last 90 days, not a forecast, on a data grid of about 10–25 km. It can miss local differences, especially in Western Ghats districts.
- District locations are approximate HQ coordinates entered by hand.
- OCR reads English and digits only. It was tested on a generated card image, not yet on many real soil cards.
- Kannada and Hindi text was machine-written and needs checking by a native speaker.
- Voice and weather need internet.
- The app has not yet been tested with real farmers.
- The recommendation does not yet consider profit, market price, water availability or crop rotation.

---

## 7. Future work

1. **Profit ranking:** combine crop suitability with mandi (Agmarknet) prices.
2. **Local data:** use Karnataka soil and crop data (ICAR, data.gov.in) instead of the generic dataset.
3. **"Why this crop"** in simple words, with voice.
4. **Taluk-level picker** for districts with big weather differences.
5. **Kannada soil-card OCR** (for example with a cloud vision API).
6. **Season planner** (Kharif / Rabi) using seasonal weather.
7. **Fertilizer recommendation** as a second output.
8. **Field test** with real farmers and improve the design from their feedback.

---

## 8. Technology used

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit, Open-Meteo API, RapidOCR, gTTS, Pillow.

## 9. Credits

- Dataset: Crop Recommendation Dataset (Kaggle, Atharva Ingle)
- Weather and geocoding: Open-Meteo (CC BY 4.0)
- Photos: add credits here

---

## 10. Phase 2: dataset integrity and leakage audit

The Phase 2 audit is implemented in `audit_dataset.py` and `evaluation/dataset_audit.py`. Run it with `python audit_dataset.py`; reports are written to `reports/phase2/` without modifying the source CSV.

On the included 2,200-row `Crop_recommendation.csv`, the audit found:

- 0 exact duplicate full rows; 0 repeated exact feature vectors; 0 conflicting crop labels for identical feature vectors.
- 0 exact feature-vector overlap and 0 exact full-row overlap between the same shuffled, stratified 80/20 train/test split used by `crop_model.py`.
- 0 train/test feature-key collisions after rounding predictors to 0, 1, 2, 3, 4 or 5 decimal places.
- Exactly 100 rows for each of 22 crop labels.
- 22 contiguous label blocks in the CSV's original row order. The model pipeline's shuffled stratified split avoids the major evaluation issue this would create for a non-shuffled split.

**Interpretation:** no direct duplicate-based leakage was detected by these checks. That is not proof of independent data collection, and it does not establish real-world reliability. The CSV has no row-level farm, region, date/season, source, soil-sampling, nutrient-unit or harvest-outcome metadata. The high random-split score should therefore be treated as performance on this dataset, not expected accuracy on real farms.

A separate input-semantics risk also needs review before deployment: `app_farmer.py` averages temperature and humidity over the last 90 days and converts the same 90-day precipitation total to a monthly average, while a public description of a commonly circulated copy of the dataset describes rainfall as a seasonal amount. Confirm the exact source definition of rainfall in the included CSV, and align the application aggregation with it before relying on recommendations. The existing report also notes that N/P/K soil-card values may not have the same units as the training dataset.

See `reports/phase2/dataset_audit.md` for the full result, CSV tables, method notes and source links. Before farmer-facing claims, collect or obtain independent field/season data, document units and label provenance, perform a grouped or temporal holdout evaluation, and seek agronomist review.
