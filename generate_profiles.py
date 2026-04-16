import pickle, random
import numpy as np

random.seed(42)
np.random.seed(42)

MALE_NAMES = [
    "Ravi Kumar", "Arjun Singh", "Karan Mehta", "Ankit Yadav", "Rohit Verma",
    "Vijay Reddy", "Suresh Pillai", "Amit Tiwari", "Manoj Pandey", "Harish Bhatt",
    "Deepak Chauhan", "Sanjay Patil", "Nikhil Joshi", "Pradeep Nair", "Rahul Das",
    "Varun Sharma", "Akash Gupta", "Vivek Rao", "Mohit Sinha", "Tarun Bhat"
]

AREAS = ["Andheri", "Bandra", "Thane", "Borivali", "Dadar",
         "Kurla", "Malad", "Goregaon", "Powai", "Navi Mumbai",
         "Vikhroli", "Mulund", "Ghatkopar", "Chembur", "Wadala",
         "Worli", "Parel", "Sion", "Matunga", "Dharavi"]

# Max delivery distance per vehicle (customer side cap)
MAX_DISTANCE = {
    "Motorcycle":      11.0,
    "Scooter":         11.0,
    "Electric Scooter":11.0,
    "Bicycle":          3.0,
}

# Realistic distance-from-restaurant range per vehicle
# Bicycles stay close, motorcycles can be further out
PARTNER_DIST_RANGE = {
    "Motorcycle":      (0.3, 4.5),
    "Scooter":         (0.3, 3.5),
    "Electric Scooter":(0.2, 3.0),
    "Bicycle":         (0.1, 1.5),
}

MAX_ACTIVE_ORDERS = 2

profiles = []
for i, name in enumerate(MALE_NAMES):
    if   i < 7:  vehicle = "Motorcycle"
    elif i < 13: vehicle = "Scooter"
    elif i < 17: vehicle = "Electric Scooter"
    else:        vehicle = "Bicycle"

    age               = random.randint(19, 42)
    rating_bands      = [3.4, 3.7, 4.0, 4.3, 4.6, 4.8, 5.0]
    rating            = round(random.choice(rating_bands) + random.uniform(-0.1, 0.1), 1)
    rating            = round(max(3.4, min(5.0, rating)), 1)
    deliveries_today  = random.randint(0, MAX_ACTIVE_ORDERS)
    total_deliveries  = random.randint(120, 2800)
    vehicle_condition = random.randint(2, 3)
    experience_months = random.randint(2, 48)
    area              = AREAS[i % len(AREAS)]

    # ── Fixed distance of this partner FROM the restaurant ────
    dmin, dmax = PARTNER_DIST_RANGE[vehicle]
    dist_from_restaurant = round(random.uniform(dmin, dmax), 1)

    vehicle_map = {"Motorcycle": 0, "Scooter": 1, "Electric Scooter": 2, "Bicycle": 3}
    vehicle_enc = vehicle_map[vehicle]

    cancellation_rate = round(random.uniform(0.01, 0.10), 2)

    clv_base = round(
        (rating / 5.0) * 40 +
        min(total_deliveries / 3000, 1) * 30 +
        min(experience_months / 48, 1) * 20 +
        (1 - cancellation_rate) * 10,
        1
    )

    quick_score     = round(clv_base * (1.2 if vehicle in ["Motorcycle","Electric Scooter"] else 0.8) * (rating / 5), 1)
    standard_score  = round(clv_base * 1.0, 1)
    scheduled_score = round(clv_base * (1.1 if vehicle == "Bicycle" else 1.0), 1)

    profiles.append({
        "id":                      i + 1,
        "name":                    name,
        "age":                     age,
        "rating":                  rating,
        "vehicle_display":         vehicle,
        "vehicle_enc":             vehicle_enc,
        "vehicle_condition":       vehicle_condition,
        "multiple_deliveries":     deliveries_today,
        "max_distance_km":         MAX_DISTANCE[vehicle],
        "dist_from_restaurant_km": dist_from_restaurant,   # ← FIXED per partner
        "area":                    area,
        "experience_months":       experience_months,
        "total_deliveries":        total_deliveries,
        "cancellation_rate":       cancellation_rate,
        "clv_base":                clv_base,
        "tier_scores": {
            "Quick":     min(quick_score, 100),
            "Standard":  min(standard_score, 100),
            "Scheduled": min(scheduled_score, 100),
        }
    })

with open("models/delivery_profiles.pkl", "wb") as f:
    pickle.dump(profiles, f)

print(f"✅ Generated {len(profiles)} delivery profiles\n")
print(f"  {'Name':<20} {'Vehicle':<16} {'Rating':<7} {'Dist→Restaurant':<17} {'Active':<8} {'CLV'}")
print("  " + "-"*75)
for p in profiles:
    print(f"  {p['name']:<20} {p['vehicle_display']:<16} ⭐{p['rating']:<5} "
          f"📍{p['dist_from_restaurant_km']:.1f} km{'':<9} "
          f"📦{p['multiple_deliveries']} active   {p['clv_base']}")
