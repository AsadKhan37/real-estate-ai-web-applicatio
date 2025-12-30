from flask import Flask, render_template, request
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = Flask(__name__)

# ---- Paths ----
model_path_joblib = r"C:/code1/models/model_rf.joblib"
csv_path = r"C:/code1/Dataset/housing.csv"

# ---- Load model & data ----
loaded_model_joblib = joblib.load(model_path_joblib)
housing = pd.read_csv(csv_path)

# ---- Features/target ----
TARGET_COL = "MEDV"  # you can change if needed
FEATURE_COLS = [c for c in housing.columns if c != TARGET_COL]

# ---- Stats for UI (min/max/mean) ----
desc = housing[FEATURE_COLS].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
stats = desc[["min", "mean", "max"]].to_dict("index")

# ---- Preprocessing ----
data_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
data_pipeline.fit(housing[FEATURE_COLS])

@app.route("/")
def index():
    return render_template(
        "index.html",
        feature_cols=FEATURE_COLS,
        stats=stats,
        prediction=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Build input row in FEATURE_COLS order
    row = []
    for col in FEATURE_COLS:
        val = request.form.get(col, "")
        if val.strip() == "":
            row.append(np.nan)
        else:
            try:
                row.append(float(val))
            except ValueError:
                row.append(np.nan)

    X = data_pipeline.transform([row])
    y = loaded_model_joblib.predict(X)[0]

    return render_template(
        "index.html",
        feature_cols=FEATURE_COLS,
        stats=stats,
        prediction=float(y)
    )

if __name__ == "__main__":
    app.run(debug=True)
