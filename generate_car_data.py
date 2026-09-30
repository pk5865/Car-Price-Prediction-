import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

random.seed(42)

companies = {
    "Maruti": ["Swift", "Alto", "Celerio", "WagonR"],
    "Hyundai": ["Creta", "i20", "Venue", "Xcent"],
    "Tata": ["Nexon", "Harrier", "Punch", "Safari"],
    "Honda": ["City", "Accord", "CR-V", "Jazz"],
    "Mahindra": ["XUV500", "Bolero", "Scorpio", "Xylo"],
}

fuel_factors = {
    "Petrol": 1.00,
    "Diesel": 1.08,
    "CNG": 0.78,
    "LPG": 0.76,
}

base_price = {
    "Maruti": {"Swift": 650000, "Alto": 420000, "Celerio": 480000, "WagonR": 500000},
    "Hyundai": {"Creta": 1050000, "i20": 760000, "Venue": 880000, "Xcent": 720000},
    "Tata": {"Nexon": 980000, "Harrier": 1350000, "Punch": 760000, "Safari": 1500000},
    "Honda": {"City": 1100000, "Accord": 1800000, "CR-V": 1900000, "Jazz": 730000},
    "Mahindra": {"XUV500": 1250000, "Bolero": 970000, "Scorpio": 1400000, "Xylo": 870000},
}

rows = []
for company, models in companies.items():
    for model in models:
        for year in range(2015, 2026):
            for fuel_type in ["Petrol", "Diesel", "CNG", "LPG"]:
                for _ in range(6):
                    kms = random.randint(3000, 180000)
                    age = 2026 - year
                    model_factor = 1.0 + (0.04 * (year - 2015))
                    mileage_factor = 1.0 - (kms / 500000)
                    fuel_factor = fuel_factors[fuel_type]

                    price = base_price[company][model] * model_factor * fuel_factor * (0.75 + (0.15 * (5 - age) / 5))
                    price *= 1.0 + random.uniform(-0.12, 0.18)
                    price *= max(0.4, mileage_factor)
                    price = max(150000, round(price / 1000) * 1000)

                    rows.append(
                        {
                            "company": company,
                            "model": model,
                            "year": year,
                            "kms_driven": kms,
                            "fuel_type": fuel_type,
                            "price": price,
                        }
                    )

# Keep dataset balanced and realistic
frame = pd.DataFrame(rows)
frame = frame.drop_duplicates().reset_index(drop=True)
frame.to_csv(ROOT / "car_data.csv", index=False)
print(f"Generated {len(frame)} rows in {ROOT / 'car_data.csv'}")
