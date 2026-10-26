# AgriSense Crop Recommendation

AgriSense is a Streamlit crop-recommendation application backed by the
existing trained Random Forest model in `crop_model.pkl`. The primary entry
point is `app.py`; `app_farmer.py` is an alternative multilingual,
voice-oriented interface with optional soil-card OCR.

## Features

- Crop ranking from seven numeric inputs: N, P, K, temperature, humidity, pH,
  and rainfall.
- Optional Open-Meteo weather lookup with no API key.
- Optional historical regional evidence from
  `data/regional/crop_yield.csv`.
- Input validation and warnings for values outside observed training ranges.
- Optional soil-card OCR in `app_farmer.py`.
- Existing model/input-semantics provenance support.

## Required repository files

Keep these paths relative to the repository root:

```text
app.py
app_farmer.py
crop_model.pkl
Crop_recommendation.csv
data/regional/crop_yield.csv
evaluation/
assets/                  # used by app_farmer.py
requirements.txt
```

The application does not retrain the model at startup. Do not replace
`crop_model.pkl` or modify the datasets as part of deployment.

## Installation

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the alternative farmer interface, which includes text-to-speech and
optional soil-card OCR:

```powershell
pip install -r requirements-farmer.txt
```

For development, tests, and model-training scripts:

```powershell
pip install -r requirements-dev.txt
```

## Run locally

Primary application:

```powershell
streamlit run app.py
```

Alternative farmer interface:

```powershell
streamlit run app_farmer.py
```

The commands work from any current directory because required model, dataset,
regional, and asset paths are resolved relative to the Python files.

## Deployment

1. Deploy the repository with the required files listed above.
2. Use Python 3.12 or a compatible supported Python 3 release.
3. Install `requirements.txt`.
4. Configure the service to run:

   ```text
   streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
   ```

   For platforms that do not provide `$PORT`, replace it with the platform's
   assigned port.
5. Keep `crop_model.pkl` and `Crop_recommendation.csv` at the repository root
   and `data/regional/crop_yield.csv` at its documented path.

Open-Meteo's public endpoints are used for geocoding and historical weather;
no API key or application secret is currently required. Weather lookup is
optional. If the network is unavailable, users can continue with manually
entered weather values; the app displays an error instead of silently
pretending that weather was fetched.

### Deploy on Render

1. Push this repository to GitHub, including `crop_model.pkl`,
   `Crop_recommendation.csv`, `data/regional/crop_yield.csv`, and the `assets/`
   directory.
2. In Render, select **New +** → **Web Service**, connect the GitHub
   repository, and select the branch to deploy.
3. Use these settings:

   | Setting | Value |
   | --- | --- |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT` |
   | Health Check Path | `/_stcore/health` |

4. Add the environment variable `PYTHON_VERSION` with value `3.12` in
   Render's Environment settings. The repository also includes a
   `.python-version` file for tools that support automatic version detection.
5. Choose the **Free** instance for testing or a paid instance for production,
   then click **Create Web Service**.

Render supplies the `$PORT` environment variable. Binding Streamlit to
`0.0.0.0` and that port is required for the service to become reachable.
After deployment, open the generated `onrender.com` URL. To deploy the
alternative farmer interface instead, use
`streamlit run app_farmer.py --server.address=0.0.0.0 --server.port=$PORT` as
the Start Command and use `pip install -r requirements-farmer.txt` as the
Build Command.

## Validation

Run the existing test suite from the repository root:

```powershell
python -m pytest tests -q
```

## Known limitations

- The model's original feature units and rainfall aggregation period are not
  fully documented, so model scores are not probabilities of crop success or
  expected yield.
- Soil-card Low/Medium/High values use a training-distribution proxy rather
  than a verified physical-unit conversion.
- Rainfall is derived from a recent weather window and its compatibility with
  the training rainfall feature remains unresolved.
- Regional evidence is historical association, not proof of suitability or
  causal benefit.
- `app_farmer.py`'s OCR depends on the optional `rapidocr-onnxruntime`
  package; if unavailable, manual soil-card entry remains available.
- Internet access is required for weather lookup and text-to-speech features;
  the core local model does not require an API key.
