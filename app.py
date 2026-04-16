from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, pandas as pd
from datetime import datetime

app = Flask(__name__)

with open("models/xgb_model.pkl",        "rb") as f: model     = pickle.load(f)
with open("models/encoders.pkl",          "rb") as f: encoders  = pickle.load(f)
with open("models/encoder_classes.pkl",   "rb") as f: classes   = pickle.load(f)
with open("models/feature_cols.pkl",      "rb") as f: feat_cols = pickle.load(f)
with open("models/delivery_profiles.pkl", "rb") as f: profiles  = pickle.load(f)
with open("models/metrics.pkl",           "rb") as f: metrics   = pickle.load(f)
with open("models/eng_meta.pkl",          "rb") as f: eng_meta  = pickle.load(f)

MAX_DELIVERY_KM   = 11.0
BICYCLE_MAX_KM    = 3.0
MAX_ACTIVE_ORDERS = 2

TIER_MULTIPLIER   = {"Quick": 1.0, "Standard": 1.25, "Scheduled": 1.55}
TIER_PRICE        = {"Quick": 49,  "Standard": 29,   "Scheduled": 15}

TRAFFIC_RANK_MAP  = eng_meta.get("traffic_rank_map", {"Low":0,"Medium":1,"High":2,"Jam":3})

# Speed (km/min) × traffic factor — bicycle barely slows in jams
BASE_SPEED = {"Motorcycle":0.55,"Scooter":0.45,"Electric Scooter":0.40,"Bicycle":0.18}
TRAFFIC_FACTOR = {
    "Motorcycle":       [1.00,0.75,0.55,0.38],
    "Scooter":          [1.00,0.78,0.58,0.42],
    "Electric Scooter": [1.00,0.80,0.62,0.48],
    "Bicycle":          [1.00,0.93,0.85,0.78],
}
ACTIVE_ORDER_DELAY = 4.5   # min per queued order

def is_eligible(p, customer_dist):
    if p["vehicle_display"] == "Bicycle" and customer_dist > BICYCLE_MAX_KM: return False
    if customer_dist > float(p.get("max_distance_km", MAX_DELIVERY_KM)):     return False
    if float(p.get("multiple_deliveries", 0)) > MAX_ACTIVE_ORDERS:           return False
    return True

def build_feature_row(profile, ctx):
    """Build the exact feature vector the model was trained on."""
    traffic_label   = ctx["traffic"]
    t_rank          = TRAFFIC_RANK_MAP.get(traffic_label, 1)
    total_dist      = float(profile["dist_from_restaurant_km"]) + float(ctx["distance_km"])
    active          = float(profile.get("multiple_deliveries", 0))
    age             = float(profile["age"])
    rating          = float(profile["rating"])
    vc              = float(profile["vehicle_condition"])
    now             = datetime.now()
    hour            = float(now.hour)
    dow             = float(now.weekday())
    is_peak         = 1.0 if (12<=now.hour<=14 or 19<=now.hour<=22) else 0.0
    is_weekend      = 1.0 if now.weekday() >= 5 else 0.0
    prep_time       = ctx.get("prep_time", 8.0)   # default 8 min kitchen prep

    row = {
        "Delivery_person_Age":     age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition":       vc,
        "multiple_deliveries":     active,
        "distance_km":             total_dist,
        # Engineered
        "prep_time":               prep_time,
        "hour_of_day":             hour,
        "is_peak_hour":            is_peak,
        "day_of_week":             dow,
        "is_weekend":              is_weekend,
        "traffic_rank":            float(t_rank),
        "dist_x_traffic":          total_dist * (t_rank + 1),
        "log_distance":            np.log1p(total_dist),
        "exp_speed_ratio":         total_dist / max(age - 17, 1),
        "rating_x_condition":      rating * vc,
    }

    # Categorical encodings
    for col, label in [
        ("Road_traffic_density", traffic_label),
        ("Type_of_vehicle",      profile["vehicle_display"]),
        ("Weatherconditions",    ctx.get("weather", "Sunny")),
        ("Festival",             ctx.get("festival", "No")),
        ("City",                 ctx.get("city",    "Urban")),
    ]:
        if col in encoders:
            try:
                row[col] = float(encoders[col].transform([label])[0])
            except:
                row[col] = float(encoders[col].transform([encoders[col].classes_[0]])[0])

    # Keep only features model knows, in correct order
    row_filtered = {k: row[k] for k in feat_cols if k in row}
    # Fill any missing with 0
    for k in feat_cols:
        if k not in row_filtered:
            row_filtered[k] = 0.0

    return pd.DataFrame([row_filtered])[feat_cols]

def predict_eta(profile, ctx):
    vehicle       = profile["vehicle_display"]
    partner_dist  = float(profile["dist_from_restaurant_km"])
    traffic_label = ctx["traffic"]
    t_idx         = TRAFFIC_RANK_MAP.get(traffic_label, 1)

    speed         = BASE_SPEED.get(vehicle, 0.45)
    factor        = TRAFFIC_FACTOR.get(vehicle, [1,0.75,0.55,0.38])[min(t_idx,3)]
    pickup_mins   = partner_dist / (speed * factor)
    queue_delay   = float(profile.get("multiple_deliveries", 0)) * ACTIVE_ORDER_DELAY

    df_row        = build_feature_row(profile, ctx)
    ml_delivery   = max(5, float(model.predict(df_row)[0])) * TIER_MULTIPLIER[ctx["tier"]]

    total         = pickup_mins + queue_delay + ml_delivery
    return round(total), round(pickup_mins), round(queue_delay), round(ml_delivery)

def priority_score(p, tier, total_eta, queue_delay):
    rating  = p["rating"]
    cancel  = p["cancellation_rate"]
    active  = p["multiple_deliveries"]
    clv     = p["tier_scores"][tier]
    load    = active / (MAX_ACTIVE_ORDERS + 1)
    eta_s   = max(0, 1 - total_eta / 120)

    if   tier == "Quick":     return round(eta_s*60 + (rating/5)*25 + (1-load)*10 + (clv/100)*5, 2)
    elif tier == "Standard":  return round((rating/5)*40 + eta_s*30 + (1-load)*15 + (1-cancel)*10 + (clv/100)*5, 2)
    else:                     return round((rating/5)*45 + (1-cancel)*25 + eta_s*15 + (1-load)*10 + (clv/100)*5, 2)

@app.route("/")
def index():
    return render_template("index.html",
        traffic_opts = classes.get("Road_traffic_density", ["Low","Medium","High","Jam"]),
        metrics      = metrics,
        max_distance = MAX_DELIVERY_KM,
    )

@app.route("/predict", methods=["POST"])
def predict():
    data          = request.json
    customer_dist = float(data["distance_km"])
    traffic       = data["traffic"]
    tier          = data.get("tier", "Standard")

    if customer_dist > MAX_DELIVERY_KM:
        return jsonify({"error":"out_of_range",
                        "message":f"Delivery not available beyond {MAX_DELIVERY_KM:.0f} km.",
                        "profiles":[],"metrics":metrics,"tier":tier})

    ctx = {"distance_km":customer_dist,"traffic":traffic,"tier":tier}

    results = []
    for p in profiles:
        if not is_eligible(p, customer_dist): continue
        total_eta, pickup_mins, queue_delay, delivery_mins = predict_eta(p, ctx)
        p_score = priority_score(p, tier, total_eta, queue_delay)
        results.append({
            "id":                  p["id"],
            "name":                p["name"],
            "age":                 p["age"],
            "rating":              p["rating"],
            "vehicle_display":     p["vehicle_display"],
            "vehicle_condition":   p["vehicle_condition"],
            "area":                p["area"],
            "max_distance_km":     p.get("max_distance_km", MAX_DELIVERY_KM),
            "dist_from_restaurant":p["dist_from_restaurant_km"],
            "experience_months":   p["experience_months"],
            "total_deliveries":    p["total_deliveries"],
            "cancellation_rate":   p["cancellation_rate"],
            "clv_score":           round(p["tier_scores"][tier],1),
            "tier_scores":         p["tier_scores"],
            "multiple_deliveries": p["multiple_deliveries"],
            "pickup_mins":         pickup_mins,
            "queue_delay":         queue_delay,
            "delivery_eta":        delivery_mins,
            "eta":                 total_eta,
            "priority_score":      p_score,
            "tier_premium":        TIER_PRICE[tier],
        })

    results.sort(key=lambda x: -x["priority_score"])
    return jsonify({"profiles":results,"metrics":metrics,
                    "tier":tier,"customer_dist":customer_dist})

if __name__ == "__main__":
    app.run(debug=True)
