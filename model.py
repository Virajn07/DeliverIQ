"""
DeliverIQ — model.py  (Feature-Engineered, All 6 LOs)
═══════════════════════════════════════════════════════════════════════
  LO1  Regression Techniques     → Linear Regression (baseline)
  LO2  Support Vector Machine    → SVR (sampled)
  LO3  Various ML Models         → LR · RF · SVR · MLP · XGBoost
  LO4  Suitable Model Selection  → XGBoost (best R²/RMSE) deployed
  LO5  Neural Network            → MLPRegressor
  LO6  Dimensionality Reduction  → PCA variance analysis
═══════════════════════════════════════════════════════════════════════
Feature Engineering applied:
  • Haversine distance (lat/lon)
  • Prep time = Time_Order_picked - Time_Orderd
  • Hour of day + is_peak_hour flag
  • Day of week + is_weekend flag
  • distance × traffic interaction
  • speed_ratio = distance / (age proxy)
  • log(distance) for skew correction
  • All original high-signal features retained
"""

import pandas as pd, numpy as np, pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import r2_score, mean_squared_error
from sklearn.linear_model     import LinearRegression
from sklearn.svm              import SVR
from sklearn.ensemble         import RandomForestRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.decomposition    import PCA
from xgboost                  import XGBRegressor

os.makedirs("models", exist_ok=True)
np.random.seed(42)
SEP  = "─" * 62
SEP2 = "═" * 62

# ══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════
print(SEP2); print("  STEP 1 — LOAD & CLEAN"); print(SEP2)
df = pd.read_csv("train.csv")
print(f"  Raw shape   : {df.shape}")
print(f"  Columns     : {df.columns.tolist()}\n")

# ── Clean target ──────────────────────────────────────────────
df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)

# ── Clean weather prefix ──────────────────────────────────────
if "Weatherconditions" in df.columns:
    df["Weatherconditions"] = df["Weatherconditions"].astype(str).str.replace("conditions ", "", regex=False).str.strip()

# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print(SEP2); print("  STEP 2 — FEATURE ENGINEERING"); print(SEP2)

# ── A) Haversine distance ─────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

for c in ["Restaurant_latitude","Restaurant_longitude",
          "Delivery_location_latitude","Delivery_location_longitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["distance_km"] = haversine(
    df["Restaurant_latitude"],  df["Restaurant_longitude"],
    df["Delivery_location_latitude"], df["Delivery_location_longitude"]
)
print("  ✅ distance_km (Haversine)")

# ── B) Time features from Order_Date + Time columns ──────────
def parse_time_col(series):
    """Parse HH:MM or HH:MM:SS strings → total minutes since midnight."""
    def to_mins(val):
        try:
            parts = str(val).strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return np.nan
    return series.apply(to_mins)

if "Time_Orderd" in df.columns and "Time_Order_picked" in df.columns:
    df["order_mins"]  = parse_time_col(df["Time_Orderd"])
    df["pickup_mins"] = parse_time_col(df["Time_Order_picked"])
    df["prep_time"]   = (df["pickup_mins"] - df["order_mins"]).clip(lower=0)
    df["hour_of_day"] = (df["order_mins"] / 60).round().astype(float)
    df["is_peak_hour"]= df["hour_of_day"].apply(lambda h: 1 if (12<=h<=14 or 19<=h<=22) else 0)
    print("  ✅ prep_time, hour_of_day, is_peak_hour")
else:
    df["prep_time"]   = 0.0
    df["hour_of_day"] = 13.0   # default: lunch hour
    df["is_peak_hour"]= 1
    print("  ⚠️  Time columns not found — using defaults")

if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
    df["day_of_week"] = df["Order_Date"].dt.dayofweek.astype(float)
    df["is_weekend"]  = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
    print("  ✅ day_of_week, is_weekend")
else:
    df["day_of_week"] = 2.0
    df["is_weekend"]  = 0
    print("  ⚠️  Order_Date not found — using defaults")

# ── C) Numeric coercions ──────────────────────────────────────
for c in ["Delivery_person_Age","Delivery_person_Ratings",
          "Vehicle_condition","multiple_deliveries","distance_km"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ── D) Interaction & derived features ────────────────────────
# Encode traffic numerically for interaction (before LabelEncoder scrambles order)
TRAFFIC_RANK = {"Low":0, "Medium":1, "High":2, "Jam":3}
if "Road_traffic_density" in df.columns:
    df["traffic_rank"] = df["Road_traffic_density"].astype(str).map(TRAFFIC_RANK).fillna(1)
    df["dist_x_traffic"] = df["distance_km"] * (df["traffic_rank"] + 1)  # interaction
    print("  ✅ traffic_rank, dist_x_traffic (distance × traffic interaction)")
else:
    df["traffic_rank"]    = 1.0
    df["dist_x_traffic"]  = df["distance_km"]

# Log distance (corrects right skew)
df["log_distance"] = np.log1p(df["distance_km"])
print("  ✅ log_distance")

# Speed proxy: distance per unit age (older riders → more experience)
df["exp_speed_ratio"] = df["distance_km"] / (df["Delivery_person_Age"].clip(lower=18) - 17)
print("  ✅ exp_speed_ratio")

# Rating × condition interaction (high rating + good vehicle = faster)
if "Vehicle_condition" in df.columns:
    df["rating_x_condition"] = df["Delivery_person_Ratings"] * df["Vehicle_condition"]
    print("  ✅ rating_x_condition")

# ══════════════════════════════════════════════════════════════
# 3. FINAL FEATURE SET
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  STEP 3 — FINAL FEATURE SET"); print(SEP2)

BASE_FEATURES = [
    # Original high-signal features
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Road_traffic_density", "Vehicle_condition",
    "Type_of_vehicle", "multiple_deliveries",
    "distance_km",
    # Kept back (do help after proper encoding)
    "Weatherconditions", "Festival", "City",
    # Engineered features
    "prep_time", "hour_of_day", "is_peak_hour",
    "day_of_week", "is_weekend",
    "traffic_rank", "dist_x_traffic",
    "log_distance", "exp_speed_ratio", "rating_x_condition",
    # Target
    "Time_taken(min)"
]

use_cols = [c for c in BASE_FEATURES if c in df.columns]
df = df[use_cols].dropna()
print(f"  Features used : {[c for c in use_cols if c != 'Time_taken(min)']}")
print(f"  Clean shape   : {df.shape}\n")

# ── Encode categoricals ───────────────────────────────────────
CAT_COLS = [c for c in ["Road_traffic_density","Type_of_vehicle",
                         "Weatherconditions","Festival","City"]
            if c in df.columns]
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

with open("models/encoders.pkl",        "wb") as f: pickle.dump(encoders, f)
with open("models/encoder_classes.pkl", "wb") as f:
    pickle.dump({c: list(e.classes_) for c,e in encoders.items()}, f)

X = df.drop("Time_taken(min)", axis=1)
y = df["Time_taken(min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
with open("models/feature_cols.pkl", "wb") as f: pickle.dump(list(X.columns), f)
print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
print(f"  Feature count: {X.shape[1]}\n")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
with open("models/scaler.pkl", "wb") as f: pickle.dump(scaler, f)

def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<32}  R²: {r2:.4f}  RMSE: {rmse:.4f}")
    return {"Model": name, "R²": round(r2,4), "RMSE": round(rmse,4)}

results = []

# ══════════════════════════════════════════════════════════════
# LO1 — LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════
print(SEP2); print("  LO1 — LINEAR REGRESSION"); print(SEP2)
lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test)))
print(f"\n  Top 5 coefficients:")
coef_df = pd.Series(lr.coef_, index=X.columns).abs().sort_values(ascending=False)
for feat in coef_df.head(5).index:
    print(f"    {feat:35s}  {lr.coef_[list(X.columns).index(feat)]:+.4f}")

# ══════════════════════════════════════════════════════════════
# LO3 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  LO3 — RANDOM FOREST"); print(SEP2)
rf = RandomForestRegressor(n_estimators=200, max_depth=14,
                           min_samples_split=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))
print(f"\n  Top 5 feature importances:")
imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1])[:5]
for feat, score in imp:
    print(f"    {feat:35s}  {'█'*int(score*50):<20}  {score:.4f}")

# ══════════════════════════════════════════════════════════════
# LO2 — SVR
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  LO2 — SVR (sampled)"); print(SEP2)
idx = np.random.choice(len(X_train_sc), size=min(5000, len(X_train_sc)), replace=False)
svr = SVR(kernel="rbf", C=100, epsilon=0.3, gamma="scale")
svr.fit(X_train_sc[idx], y_train.iloc[idx])
results.append(evaluate("SVR (RBF)", y_test, svr.predict(X_test_sc)))

# ══════════════════════════════════════════════════════════════
# LO5 — MLP NEURAL NETWORK
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  LO5 — MLP NEURAL NETWORK"); print(SEP2)
mlp = MLPRegressor(hidden_layer_sizes=(256,128,64), activation="relu",
                   solver="adam", learning_rate_init=0.001,
                   max_iter=400, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20,
                   random_state=42)
mlp.fit(X_train_sc, y_train)
results.append(evaluate("MLP Neural Network", y_test, mlp.predict(X_test_sc)))
print(f"  Stopped at iteration: {mlp.n_iter_}")

# ══════════════════════════════════════════════════════════════
# LO3+LO4 — XGBOOST (Deployed)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  LO3+LO4 — XGBOOST (Deployed Model)"); print(SEP2)
xgb = XGBRegressor(
    n_estimators    = 500,
    learning_rate   = 0.03,
    max_depth       = 7,
    subsample       = 0.85,
    colsample_bytree= 0.75,
    min_child_weight= 3,
    reg_alpha       = 0.1,
    reg_lambda      = 1.5,
    gamma           = 0.05,
    random_state    = 42,
    n_jobs          = -1,
    verbosity       = 0
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)
xgb_preds = xgb.predict(X_test)
xgb_res   = evaluate("XGBoost", y_test, xgb_preds)
results.append(xgb_res)

with open("models/xgb_model.pkl","wb") as f: pickle.dump(xgb, f)
with open("models/metrics.pkl",  "wb") as f:
    pickle.dump({"R2": xgb_res["R²"], "RMSE": xgb_res["RMSE"]}, f)

print(f"\n  Top 5 XGBoost feature importances:")
xi = sorted(zip(X.columns, xgb.feature_importances_), key=lambda x: -x[1])[:5]
for feat, score in xi:
    print(f"    {feat:35s}  {'█'*int(score*60):<20}  {score:.4f}")

# ══════════════════════════════════════════════════════════════
# LO6 — PCA
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  LO6 — PCA DIMENSIONALITY REDUCTION"); print(SEP2)
n_feat   = X_train_sc.shape[1]
pca_full = PCA(n_components=n_feat, random_state=42).fit(X_train_sc)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
n_95     = int(np.searchsorted(cum_var, 0.95)) + 1
n_99     = int(np.searchsorted(cum_var, 0.99)) + 1

print(f"  Original features      : {n_feat}")
print(f"  Components for 95% var : {n_95}")
print(f"  Components for 99% var : {n_99}")
print(f"\n  Variance per component:")
for i,(v,cv) in enumerate(zip(pca_full.explained_variance_ratio_, cum_var)):
    bar = "█"*int(v*60)
    print(f"    PC{i+1:02d}  {bar:<20}  {v*100:5.2f}%  cum:{cv*100:.1f}%")

pca_95  = PCA(n_components=n_95, random_state=42)
xgb_pca = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=7,
                        subsample=0.85, colsample_bytree=0.75,
                        random_state=42, n_jobs=-1, verbosity=0)
xgb_pca.fit(pca_95.fit_transform(X_train_sc), y_train)
pca_res = evaluate(f"XGBoost+PCA({n_95}comps)", y_test, xgb_pca.predict(pca_95.transform(X_test_sc)))
results.append(pca_res)

dim_pct = (1 - n_95/n_feat)*100
print(f"\n  Dimensionality reduced {dim_pct:.1f}% | R² change: {abs(xgb_res['R²']-pca_res['R²']):.4f}")

with open("models/pca_analysis.pkl","wb") as f:
    pickle.dump({"n_original":n_feat,"n_95pct":n_95,"n_99pct":n_99,
                 "explained_var":pca_full.explained_variance_ratio_.tolist(),
                 "cumulative_var":cum_var.tolist(),
                 "xgb_raw_r2":xgb_res["R²"],"xgb_pca_r2":pca_res["R²"],
                 "dim_reduction_pct":round(dim_pct,1)}, f)

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}"); print("  MODEL COMPARISON"); print(SEP2)
res_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
print(f"\n  {'Model':<34}  {'R²':>7}  {'RMSE':>8}  Bar")
print(f"  {SEP}")
for _,r in res_df.iterrows():
    bar = "█"*int(r["R²"]*35)
    tag = " ← DEPLOYED" if r["Model"]=="XGBoost" else ""
    print(f"  {r['Model']:<34}  {r['R²']:>7.4f}  {r['RMSE']:>8.4f}  {bar}{tag}")

with open("models/model_comparison.pkl","wb") as f:
    pickle.dump(res_df.to_dict("records"), f)

# ══════════════════════════════════════════════════════════════
# SAVE ENGINEERED FEATURE NAMES FOR app.py
# ══════════════════════════════════════════════════════════════
# Save which time/interaction features we engineered so app.py knows what to build
eng_meta = {
    "has_prep_time":     "prep_time"       in X.columns,
    "has_hour_of_day":   "hour_of_day"     in X.columns,
    "has_is_peak":       "is_peak_hour"    in X.columns,
    "has_day_of_week":   "day_of_week"     in X.columns,
    "has_is_weekend":    "is_weekend"      in X.columns,
    "has_traffic_rank":  "traffic_rank"    in X.columns,
    "has_dist_traffic":  "dist_x_traffic"  in X.columns,
    "has_log_dist":      "log_distance"    in X.columns,
    "has_exp_speed":     "exp_speed_ratio" in X.columns,
    "has_rating_cond":   "rating_x_condition" in X.columns,
    "traffic_rank_map":  TRAFFIC_RANK,
}
with open("models/eng_meta.pkl","wb") as f: pickle.dump(eng_meta, f)

# Run generate_profiles
import importlib.util
spec = importlib.util.spec_from_file_location("gen","generate_profiles.py")
if spec:
    try:
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
    except Exception as e: print(f"  Profiles: {e}")

print(f"\n{SEP2}")
print(f"  ✅ XGBoost R²: {xgb_res['R²']}  RMSE: {xgb_res['RMSE']} min")
print(f"  Feature count: {X.shape[1]}  |  Train rows: {X_train.shape[0]}")
print(SEP2)
