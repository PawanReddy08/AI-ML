# Retail Weekly Sales Prediction

End-to-end ML project: predict a store's weekly sales using time-series features,
promotions, holidays and economic indicators.

## Pipeline
1. **Data** – 10 stores x 209 weeks (synthetic, Walmart-style; swap in a Kaggle dataset easily)
2. **Cleaning** – median imputation for missing temperatures
3. **Feature engineering** – year/month/week, lag-1, lag-4, 4-week rolling mean (leakage-safe)
4. **Time-based train/test split** – last 26 weeks held out (no random shuffling)
5. **Models** – Naive baseline, Linear Regression, Random Forest, Gradient Boosting (sklearn Pipelines)
6. **Evaluation** – MAE, RMSE, R²
7. **Deployment** – Streamlit web app

## Results (test set, last 26 weeks)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive baseline | 4969 | 7472 | 0.774 |
| Linear Regression | 3512 | 4931 | 0.901 |
| Random Forest | 1666 | 2452 | 0.976 |
| **Gradient Boosting** | **1539** | **2120** | **0.982** |

## Run it
```bash
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

## Next steps
- Use the real Kaggle Walmart / Rossmann dataset
- Hyperparameter tuning with TimeSeriesSplit + GridSearchCV
- Try XGBoost / LightGBM

## Demo
![App](images/app.png)

## Results
![Model comparison](images/model_comparison.png)
![Actual vs predicted](images/actual_vs_predicted.png)
![Feature importance](images/feature_importance.png)