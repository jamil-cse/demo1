"""Train the diabetes-risk model and save the files used by the Streamlit app.

Run:  python train_model.py
Data: Pima Indians Diabetes (UCI / Kaggle), 768 women, 8 measurements, downloaded once into ./data
"""
import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNS[:-1]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42

# ---- 1. load the real data
Path("data").mkdir(exist_ok=True)
path = Path("data/pima.csv")
if not path.exists():
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, path)
df = pd.read_csv(path, header=None, names=COLUMNS)
df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)   # a glucose of 0 is impossible = missing
print("Rows:", len(df), "| diabetic:", int(df.Outcome.sum()), f"({df.Outcome.mean():.1%})")

X, y = df[FEATURES], df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("model", model)])


# ---- 2. compare models with 5-fold cross-validation
candidates = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {n: float(cross_val_score(p, X_train, y_train, cv=cv, scoring="roc_auc").mean()) for n, p in candidates.items()}
best_name = max(cv_scores, key=cv_scores.get)
print("CV ROC-AUC:", {k: round(v, 3) for k, v in cv_scores.items()}, "-> best:", best_name)

# ---- 3. evaluate on the held-out test set
best = candidates[best_name].fit(X_train, y_train)
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED)
metrics = {
    "model": best_name,
    "cv_roc_auc": cv_scores,
    "test_roc_auc": float(roc_auc_score(y_test, proba)),
    "test_recall": float(recall_score(y_test, pred)),
    "test_f1": float(f1_score(y_test, pred)),
    "test_accuracy": float((pred == y_test).mean()),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "importance": dict(zip(FEATURES, map(float, imp.importances_mean))),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "sklearn_version": sklearn.__version__,
}
print({k: round(v, 3) for k, v in metrics.items() if k.startswith("test_")})

# ---- 4. retrain on ALL the data and save
final = pipe(best.named_steps["model"].__class__(**best.named_steps["model"].get_params())).fit(X, y)
ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())} for c in FEATURES}
metrics["ranges"] = ranges
joblib.dump(final, "model.joblib")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved model.joblib and metrics.json | scikit-learn", sklearn.__version__)
