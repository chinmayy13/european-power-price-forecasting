
from src.data_ingestion import load_data
from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.forecast import forecast_price
from src.llm_explanation import (
    explain_feature_importance,
    interpret_trading_signal,
    generate_market_report
)
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import pandas as pd



# load data

df = load_data()

print(df.columns)


# basic cleaning and qa

df = clean_data(df)

qa_report = f"""
Dataset QA Report

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Missing values:
{df.isnull().sum()}

Basic statistics:
{df.describe()}
"""

with open("qa_output/data_quality_report.txt", "w") as f:
    f.write(qa_report)

# create features (lags, net load, time features)
df = create_features(df)


# train model and evaluate
model, X_test, y_test = train_model(df)



# generate forecast
preds = forecast_price(model, X_test)


# evluate model performance
mae = mean_absolute_error(y_test, preds)

print("\nForecast sample:")
print(preds[:10])

print(f"\nMAE: {mae:.2f} €/MWh")


# plot forecast vs actual
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(preds[:200], label="Forecast")
plt.legend()
plt.title("Day Ahead Power Price Forecast")
plt.xlabel("Time")
plt.ylabel("€/MWh")

plt.savefig("figures/forecast_vs_actual.png")
plt.show()
plt.close()


#visualize net load
plt.figure(figsize=(10,5))
plt.plot(df["net_load"][:500])
plt.title("Net Load (Load - Wind - Solar)")
plt.xlabel("Time")

plt.savefig("figures/net_load.png")
plt.close()


#compare forecast with market price to generate signal
market_price = y_test.iloc[-1]
forecast_price = preds[-1]

if forecast_price > market_price:
    signal = "LONG POWER"
else:
    signal = "SHORT POWER"

print(f"\nMarket price: {market_price:.2f} €/MWh")
print(f"Forecast price: {forecast_price:.2f} €/MWh")
print("Trading signal:", signal)

# generate AI-based trading insight
trading_ai = interpret_trading_signal(
    forecast_price,
    market_price
)

print("\nAI Trading Insight:")
print(trading_ai)


# generate short market report
market_report = generate_market_report(
    forecast_price,
    market_price,
    signal
)

print("\nAI Market Report:")
print(market_report)


# check which features drive the model
importance = model.feature_importances_

features = X_test.columns

imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nFeature Importance Ranking:")
print(imp_df.to_string(index=False))

# generate AI explanation for feature importance
feature_explanation = explain_feature_importance(imp_df)

print("\nAI Feature Importance Insight:")
print(feature_explanation)



# plot feature importance
plt.figure(figsize=(8,5))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()

plt.savefig("figures/feature_importance.png")
plt.close()


# save predictions
submission = pd.DataFrame({
    "id": range(len(preds)),
    "y_pred": preds
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully.")



print("\nPipeline execution completed successfully.")