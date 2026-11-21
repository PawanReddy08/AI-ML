"""
Creates a realistic synthetic retail dataset (Walmart-style weekly sales).
Use this to run the project instantly.
To use REAL data, download "Walmart Recruiting - Store Sales Forecasting" or
"Rossmann Store Sales" from Kaggle and skip this file (see README).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2021-01-03", "2024-12-29", freq="W-SUN")
stores = range(1, 11)
store_size = {s: rng.integers(80_000, 220_000) for s in stores}      # sq ft
store_type = {s: rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2]) for s in stores}

rows = []
for s in stores:
    base = store_size[s] / 220_000 * 50_000
    for i, d in enumerate(dates):
        week = d.isocalendar().week
        # seasonality: Nov/Dec spike, mild summer bump
        season = 1 + 0.35 * (week >= 47) + 0.15 * (week == 52) + 0.05 * np.sin(2 * np.pi * week / 52)
        holiday = int(week in (6, 22, 36, 47, 52))
        temp = 55 + 25 * np.sin(2 * np.pi * (week - 12) / 52) + rng.normal(0, 4)
        fuel = 3.0 + 0.002 * i + rng.normal(0, 0.1)
        promo = int(rng.random() < 0.25)
        cpi = 210 + 0.05 * i + rng.normal(0, 0.5)
        unemp = 7.5 - 0.01 * i + rng.normal(0, 0.2)
        trend = 1 + 0.0007 * i
        sales = (base * season * trend
                 * (1 + 0.12 * holiday + 0.18 * promo)
                 * (1 - 0.002 * (fuel - 3) * 10)
                 * (1.1 if store_type[s] == "A" else 1.0 if store_type[s] == "B" else 0.9)
                 + rng.normal(0, base * 0.04))
        rows.append([s, store_type[s], store_size[s], d, holiday, round(temp, 2),
                     round(fuel, 3), promo, round(cpi, 2), round(unemp, 2), round(sales, 2)])

df = pd.DataFrame(rows, columns=["store", "store_type", "store_size", "date", "is_holiday",
                                 "temperature", "fuel_price", "promotion", "cpi",
                                 "unemployment", "weekly_sales"])
# Add a little messiness so you practise cleaning
mask = rng.random(len(df)) < 0.01
df.loc[mask, "temperature"] = np.nan
df.to_csv("data/retail_sales.csv", index=False)
print("Saved data/retail_sales.csv", df.shape)
