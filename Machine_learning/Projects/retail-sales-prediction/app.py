"""Streamlit demo: enter store details -> get predicted weekly sales.
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Sales Predictor", page_icon="🛒")
st.title("🛒 Retail Weekly Sales Predictor")

model = joblib.load("models/best_model.joblib")
df = pd.read_csv("data/retail_sales.csv")
sizes = df.groupby("store")[["store_type", "store_size"]].first()

col1, col2 = st.columns(2)
store = col1.selectbox("Store", sorted(sizes.index))
week = col1.slider("Week of year", 1, 52, 47)
year = col1.selectbox("Year", [2025, 2026])
promo = col2.checkbox("Promotion running?")
holiday = col2.checkbox("Holiday week?")
temp = col2.slider("Temperature (°F)", 10, 100, 60)
fuel = col2.number_input("Fuel price ($)", 2.5, 5.0, 3.5)
lag_1 = col1.number_input("Last week's sales ($)", 10_000, 150_000, 50_000)
lag_4 = col1.number_input("Sales 4 weeks ago ($)", 10_000, 150_000, 48_000)
avg4 = col2.number_input("Avg sales, last 4 weeks ($)", 10_000, 150_000, 49_000)

if st.button("Predict"):
    row = pd.DataFrame([{
        "store_size": sizes.loc[store, "store_size"], "is_holiday": int(holiday),
        "temperature": temp, "fuel_price": fuel, "promotion": int(promo),
        "cpi": 220.0, "unemployment": 6.5, "year": year,
        "month": min(12, (week - 1) // 4 + 1), "week": week,
        "lag_1": lag_1, "lag_4": lag_4, "rolling_mean_4": avg4,
        "store": store, "store_type": sizes.loc[store, "store_type"]}])
    st.success(f"Predicted weekly sales: ${model.predict(row)[0]:,.0f}")
