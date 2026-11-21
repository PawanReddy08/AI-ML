"""End-to-end training: load -> clean -> features -> train -> evaluate -> save."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- 1. Load ----------
df = pd.read_csv("data/retail_sales.csv", parse_dates=["date"])
df = df.sort_values(["store", "date"]).reset_index(drop=True)
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------- 2. Feature engineering ----------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week.astype(int)
# lag features: what were this store's sales 1 and 4 weeks ago? (no future leakage)
df["lag_1"] = df.groupby("store")["weekly_sales"].shift(1)
df["lag_4"] = df.groupby("store")["weekly_sales"].shift(4)
df["rolling_mean_4"] = (df.groupby("store")["weekly_sales"]
                          .transform(lambda s: s.shift(1).rolling(4).mean()))
df = df.dropna(subset=["lag_1", "lag_4", "rolling_mean_4"]).reset_index(drop=True)

num_cols = ["store_size", "is_holiday", "temperature", "fuel_price", "promotion", "cpi",
            "unemployment", "year", "month", "week", "lag_1", "lag_4", "rolling_mean_4"]
cat_cols = ["store", "store_type"]
X, y = df[num_cols + cat_cols], df["weekly_sales"]

# ---------- 3. Time-based split (NOT random: we predict the future) ----------
cutoff = df["date"].max() - pd.Timedelta(weeks=26)
train_idx, test_idx = df["date"] <= cutoff, df["date"] > cutoff
X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (last 26 weeks)")

# ---------- 4. Preprocessing pipeline ----------
def make_pipeline(model, scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])

models = {
    "Linear Regression": make_pipeline(LinearRegression(), scale=True),
    "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    "Gradient Boosting": make_pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                                   max_depth=3, random_state=42)),
}

# ---------- 5. Train + evaluate ----------
# Baseline: "next week = last week". Any real model must beat this.
base_pred = X_test["lag_1"]
results = {"Naive baseline (last week)": {
    "MAE": mean_absolute_error(y_test, base_pred),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, base_pred))),
    "R2": r2_score(y_test, base_pred)}}
preds = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    preds[name] = p
    results[name] = {"MAE": mean_absolute_error(y_test, p),
                     "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
                     "R2": r2_score(y_test, p)}

res_df = pd.DataFrame(results).T.round(3)
print("\n=== Model comparison (test set) ===\n", res_df)

# ---------- 6. Save best model ----------
best_name = res_df.drop("Naive baseline (last week)")["RMSE"].idxmin()
print("\nBest model:", best_name)
joblib.dump(models[best_name], "models/best_model.joblib")
res_df.to_csv("outputs/model_comparison.csv")
json.dump({"best_model": best_name, "num_cols": num_cols, "cat_cols": cat_cols},
          open("models/meta.json", "w"))

# ---------- 7. Plots ----------
plt.figure(figsize=(8, 4))
res_df["RMSE"].plot(kind="barh", color="steelblue")
plt.title("Model comparison (RMSE, lower is better)")
plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=120); plt.close()

# actual vs predicted for one store
s1 = X_test["store"] == 1
dates_s1 = df.loc[X_test.index[s1], "date"]
plt.figure(figsize=(10, 4))
plt.plot(dates_s1, y_test[s1], label="Actual", marker="o")
plt.plot(dates_s1, preds[best_name][s1.values], label=f"Predicted ({best_name})", marker="x")
plt.title("Store 1: actual vs predicted weekly sales (last 26 weeks)")
plt.legend(); plt.xticks(rotation=30); plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=120); plt.close()

# feature importance (tree models)
final = models[best_name]
if hasattr(final["model"], "feature_importances_"):
    names = final["pre"].get_feature_names_out()
    imp = pd.Series(final["model"].feature_importances_, index=names).sort_values().tail(10)
    plt.figure(figsize=(8, 5)); imp.plot(kind="barh", color="darkorange")
    plt.title("Top 10 feature importances"); plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=120); plt.close()
print("Saved model + plots.")
