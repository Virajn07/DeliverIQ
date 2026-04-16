<div align="center">

# 🛵 DeliverIQ

### ML-Powered Food Delivery Time Estimator & Intelligent Partner Dispatch

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Deployed-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**CSC604 — Machine Learning Lab &nbsp;|&nbsp; TE Semester VI**

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [Models](#-ml-models--lo-coverage) · [Setup](#-setup) · [Usage](#-usage) · [Results](#-results)

---

![DeliverIQ Dispatch Console](https://img.shields.io/badge/UI-Dark%20Dispatch%20Console-1B2A4A?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI0Y1QTYyMyIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==)

</div>

---

## 📌 Overview

**DeliverIQ** is a full-stack machine learning application that predicts food delivery time and intelligently dispatches delivery partners based on real-time conditions. Built on a dataset of **45,000+ Indian food delivery records**, it goes beyond simple regression — it models a realistic two-leg journey, applies traffic-aware physics per vehicle type, and scores partners via a composite priority function that adapts to the customer's delivery tier.

> **Deployed Model:** XGBoost Regressor &nbsp;|&nbsp; **R² ≈ 0.80** &nbsp;|&nbsp; **RMSE ≈ 3.2 min**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **ML ETA Prediction** | XGBoost trained on 15 engineered features for accurate delivery time estimation |
| 🏍️ **Traffic-Aware Routing** | Vehicle-specific speed degradation matrices — bicycles beat motorcycles in gridlock |
| ⚡ **Delivery Tiers** | Quick / Standard / Scheduled with dynamic pricing and priority reweighting |
| 📊 **Priority Scoring** | Composite score per partner: ETA + Rating + Active Load + CLV + Cancellation Rate |
| 👤 **Partner Profiles** | View full stats, tier suitability bars, and ETA breakdown per delivery partner |
| 🔄 **Live Auto-Update** | Every condition change triggers instant re-ranking via async fetch — no page reload |
| 💎 **Premium UI** | Dark dispatch console with animated ETA rings, staggered card animations |
| 📈 **All 6 LOs Covered** | LR · SVR · RF · MLP · XGBoost · PCA all implemented and compared |

---

## 🏗️ Architecture

```
deliveriq/
│
├── model.py                  # Training pipeline (all 6 LOs)
├── generate_profiles.py      # Synthetic delivery partner generator
├── app.py                    # Flask server — /predict REST endpoint
│
├── templates/
│   └── index.html            # Live dispatch console UI
│
├── models/                   # Auto-created on training
│   ├── xgb_model.pkl         # Deployed XGBoost model
│   ├── encoders.pkl          # LabelEncoders for categorical features
│   ├── encoder_classes.pkl   # Classes per encoder (for UI dropdowns)
│   ├── feature_cols.pkl      # Ordered feature list (training ↔ inference parity)
│   ├── scaler.pkl            # StandardScaler (SVR / MLP)
│   ├── delivery_profiles.pkl # 20 synthetic partner profiles
│   ├── metrics.pkl           # R² and RMSE of deployed model
│   ├── eng_meta.pkl          # Engineered feature metadata
│   ├── model_comparison.pkl  # All model results for report
│   └── pca_analysis.pkl      # PCA variance analysis
│
└── train.csv                 # Kaggle dataset (place here before training)
```

### Request Flow

```
Browser (condition change)
    │
    ▼  POST /predict  {traffic, distance_km, tier}
Flask app.py
    │
    ├─► For each eligible partner:
    │       ├─ Leg 1: pickup_mins = dist_from_restaurant ÷ (speed × traffic_factor)
    │       ├─ Leg 2: queue_delay = active_orders × 4.5 min
    │       └─ Leg 3: delivery_mins = XGBoost.predict(15 features) × tier_multiplier
    │
    ├─► Priority score computed per tier
    ├─► Partners sorted by score descending
    └─► JSON array returned → cards re-render with animated ETA rings
```

---

## 🤖 ML Models & LO Coverage

### Feature Engineering (15 Features)

| Feature | Source | Signal |
|---|---|---|
| `distance_km` | Haversine(GPS lat/lon) | #1 predictor — ~38% importance |
| `prep_time` | `pickup_time − order_time` | Kitchen delay — +0.10 R² lift |
| `hour_of_day` | `Time_Orderd` | Peak hour demand signal |
| `is_peak_hour` | `12–14h` or `19–22h` | Lunch/dinner rush flag |
| `day_of_week` | `Order_Date` | Weekend vs. weekday pattern |
| `is_weekend` | `day_of_week ≥ 5` | 15–25% slower on weekends |
| `traffic_rank` | Traffic ordinal (0–3) | Numeric encoding for interactions |
| `dist_x_traffic` | `distance × (traffic_rank+1)` | Distance under congestion |
| `log_distance` | `log1p(distance_km)` | Corrects right skew |
| `exp_speed_ratio` | `distance ÷ (age−17)` | Experienced rider efficiency |
| `rating_x_condition` | `rating × vehicle_condition` | High performer × good vehicle |
| `Delivery_person_Age` | Raw | Experience proxy |
| `Delivery_person_Ratings` | Raw | Historical performance |
| `Vehicle_condition` | Raw | Equipment quality (0–3) |
| `multiple_deliveries` | Raw | Current active order load |

### Model Comparison

| Model | LO | R² | RMSE (min) |
|---|---|---|---|
| Linear Regression | LO1 | ~0.45 | ~7.5 |
| SVR (RBF kernel) | LO2 | ~0.58 | ~6.4 |
| Random Forest | LO3 | ~0.72 | ~5.1 |
| MLP Neural Network | LO5 | ~0.65 | ~5.8 |
| **XGBoost** ✅ **Deployed** | **LO3+LO4** | **~0.80** | **~3.2** |
| XGBoost + PCA | LO6 | ~0.78 | ~3.5 |

> XGBoost selected as deployed model — best R² and RMSE across all comparisons.

### Delivery Tier System

| Tier | ETA Multiplier | Premium | Ranking Priority |
|---|---|---|---|
| ⚡ Quick | 1.00× | +₹49 | Speed first (60% weight on ETA) |
| 🛵 Standard | 1.25× | +₹29 | Balanced — Rating (40%) + ETA (30%) |
| 🕐 Scheduled | 1.55× | +₹15 | Reliability first (45% rating, 25% cancel rate) |

---

## ⚙️ Setup

### Prerequisites

```bash
Python 3.9+
pip
```

### 1. Clone the repository

```bash
git clone https://github.com/Virajn07/DeliverIQ.git
cd deliveriq
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost flask
```

### 3. Download the dataset

Download `train.csv` from Kaggle and place it in the project root:

> 📦 [Food Delivery Dataset — gauravmalik26](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)

```
deliveriq/
└── train.csv   ← place here
```

### 4. Train models

```bash
python model.py
```

This will:
- Load and clean the dataset
- Engineer all 15 features
- Train LR, SVR, RF, MLP, XGBoost (all 6 LOs)
- Run PCA dimensionality reduction analysis
- Save all model artifacts to `models/`
- Generate 20 synthetic delivery partner profiles

Expected output:
```
══════════════════════════════════════════════════════════════
  ✅ Done  |  XGBoost R²: 0.80  RMSE: 3.2 min
  Feature count: 15  |  Train rows: ~36,000
══════════════════════════════════════════════════════════════
```

### 5. Launch the app

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🚀 Usage

1. **Set Conditions** in the sidebar — select traffic density and drag the distance slider
2. **Choose a Delivery Tier** — Quick, Standard, or Scheduled
3. **Browse Partners** — ranked cards update instantly showing ETA breakdown per partner
4. **View Profile** — click "View Profile →" on any card for full stats, tier suitability, and ETA details
5. **Assign** — click "Assign →" or "Assign This Partner →" from the modal to dispatch

### ETA Breakdown (per card)

```
🏃 Xm pickup  +  ⏳ Xm queue  +  🛵 Xm delivery  =  N min total
```

- **Pickup leg** — partner distance to restaurant ÷ (vehicle speed × traffic factor)
- **Queue delay** — active orders × 4.5 min handoff time
- **Delivery leg** — XGBoost ML prediction × tier multiplier

---

## 📊 Results

### Why XGBoost Wins on This Dataset

```
Feature Importances (XGBoost):
  distance_km          ████████████████████  38.2%
  dist_x_traffic       ██████████            19.4%
  prep_time            ████████              15.1%
  hour_of_day          █████                  9.3%
  traffic_rank         ████                   7.8%
  log_distance         ██                     4.1%
  Delivery_person_Ratings ██                  3.6%
  (others)                                    2.5%
```

### Traffic × Vehicle Insight

In **Jam** traffic for a **2 km** order:

| Partner | Vehicle | Pickup Time | Queue | Total |
|---|---|---|---|---|
| Partner A | Motorcycle (2 orders) | 13.5 min | 9 min | **22.5 min** |
| Partner B | Bicycle (0 orders) | 14.3 min | 0 min | **14.3 min** |

> ✅ Bicycle wins — DeliverIQ correctly identifies and recommends Partner B

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | Kaggle — gauravmalik26/food-delivery-dataset |
| Records | 45,593 deliveries |
| Coverage | Multiple Indian cities |
| Target | `Time_taken(min)` — actual delivery time |
| Raw Features | 19 columns including GPS coordinates, timestamps, demographics |
| Engineered Features | 15 (after Haversine, time extraction, interaction terms) |

---

## 📚 Learning Outcomes

```
LO1  ── Linear Regression          Interpretable baseline with feature coefficients
LO2  ── Support Vector Machine     SVR with RBF kernel for non-linear regression
LO3  ── Multiple ML Models         LR · RF · SVR · MLP · XGBoost trained & compared
LO4  ── Model Selection            XGBoost selected on best R² / RMSE, justified
LO5  ── Neural Network             MLPRegressor 256→128→64 with Adam + early stopping
LO6  ── Dimensionality Reduction   PCA variance analysis, 95% threshold comparison
```

---

## 🔮 Future Works

- **Live traffic API** — replace static dropdown with Google Maps real-time data
- **LightGBM** — 3× faster training, expected R² ≥ 0.82
- **Order batching** — group nearby Scheduled orders for single-partner dispatch
- **SHAP explainability** — per-prediction feature attribution for transparency
- **Docker + MLflow** — containerised deployment with model versioning and A/B testing
- **LSTM models** — capture temporal rush-hour patterns across sequential orders

---

## 🧑‍💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

</div>

---

## 📄 References

1. Gaurav Malik — [Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset), Kaggle, 2022
2. Areta & Yalcinkaya — *A Comparative Analysis of ML Models for Food Delivery Time Prediction*, 2024 — XGBoost R²=0.82
3. Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System*, ACM KDD, 2016
4. Arxiv 2503.15177 — *Food Delivery Time Prediction in Indian Cities Using ML Models*, 2025

---

<div align="center">

**CSC604 — Machine Learning Lab &nbsp;|&nbsp; TE Semester VI**

Made with 🧠 + ☕ &nbsp;·&nbsp; DeliverIQ v3.0

</div>
