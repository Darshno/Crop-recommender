# pip install pandas scikit-learn xgboost shap joblib matplotlib seaborn
# Dataset: Kaggle "Crop Recommendation Dataset" (atharvaingle) -> Crop_recommendation.csv
# Columns: N, P, K, temperature, humidity, ph, rainfall, label

import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap

# 1. Load
df = pd.read_csv("Crop_recommendation.csv")
print(df.shape, df.isnull().sum().sum(), "missing")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]
le = LabelEncoder()
y = le.fit_transform(df["label"])

# 2. Split (stratified)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Train + compare
models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             eval_metric="mlogloss", random_state=42),
}
best_name, best_model, best_acc = None, None, 0
for name, m in models.items():
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    print(f"{name}: CV={cv:.4f}  Test={acc:.4f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, m, acc
print("Best:", best_name)

# 4. Evaluate
pred = best_model.predict(X_te)
print(classification_report(y_te, pred, target_names=le.classes_))
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_, cmap="Greens")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# 5. Feature importance (quick)
imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", title="Feature importance"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150); plt.close()

# 6. SHAP (explainability)
# 6. SHAP (explainability)
explainer = shap.TreeExplainer(best_model)
Xs = X_te.iloc[:200]
sv = explainer.shap_values(Xs)          # shape: (200, 7, 22) in new SHAP

# Overall: which features matter most across all crops
shap.summary_plot(sv, Xs, plot_type="bar", class_names=le.classes_, show=False)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150); plt.close()

# Per crop: e.g. why "rice"
ci = list(le.classes_).index("rice")
shap.summary_plot(sv[:, :, ci], Xs, show=False)
plt.tight_layout(); plt.savefig("shap_rice.png", dpi=150); plt.close()

# 7. Save
joblib.dump({"model": best_model, "encoder": le, "features": FEATURES}, "crop_model.pkl")

# 8. Top-3 prediction function (use this in Streamlit)
def recommend(N, P, K, temperature, humidity, ph, rainfall, k=3):
    bundle = joblib.load("crop_model.pkl")
    row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=bundle["features"])
    probs = bundle["model"].predict_proba(row)[0]
    top = np.argsort(probs)[::-1][:k]
    return [(bundle["encoder"].classes_[i], round(float(probs[i]) * 100, 1)) for i in top]

if __name__ == "__main__":
    print(recommend(90, 42, 43, 20.8, 82, 6.5, 202))  # expect rice
