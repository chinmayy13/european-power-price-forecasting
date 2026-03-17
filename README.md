# European Power Fair Value Forecasting Prototype

**Author:** Chinmay Kumar  
**Email:** chinmayykumarr@gmail.com

## Overview

This project builds a simple pipeline to forecast day-ahead electricity prices for the German power market (DE-LU) and translate those forecasts into a basic trading signal.

The idea was not just to build a model, but to connect price forecasting with how it might actually be used in an energy trading or analytics workflow.

## What the pipeline does

- loads and cleans power market data
- performs basic data quality checks
- creates features (lagged prices, net load, time features)
- trains an XGBoost model for forecasting
- compares forecast with market price to estimate fair value
- generates a simple LONG / SHORT signal
- produces a short AI-based explanation of the results

## Model Performance

MAE: ~2.27 €/MWh

The model improves over a simple persistence baseline and captures short-term price behavior reasonably well.

## How to run

```bash
pip install -r requirements.txt
python main.py
```
