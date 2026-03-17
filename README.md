# European Power Fair Value Forecasting Prototype


## Overview

This project builds a simple pipeline to forecast day-ahead electricity prices for the German power market (DE-LU) and translate those forecasts into a basic trading signal.

The goal was not just to build a model, but to connect price forecasting with how it might be used in a real energy trading or analytics workflow.

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

## Setup & Run

Install dependencies:

`pip install -r requirements.txt`

(Optional) To enable the AI explanation module, set your OpenAI API key:

`export OPENAI_API_KEY="your_api_key_here"`

Run the pipeline:

`python main.py`

## Outputs

Running the pipeline generates:

- `submission.csv` (forecast output)
- plots in the `figures/` folder
- QA report in `qa_output/data_quality_report.txt`
- AI logs in `qa_output/llm_log.txt`

## Notes

- Raw dataset is not included due to size; data can be downloaded from Open Power System Data
- This is a prototype and not a production trading model
- The model relies heavily on lagged prices and uses actual generation instead of forecasted fundamentals
