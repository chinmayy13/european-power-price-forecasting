import pandas as pd

def load_data():

    path = "data/raw/time_series_60min_singleindex.csv"

    df = pd.read_csv(path)

    columns = [
        "utc_timestamp",
        "DE_LU_price_day_ahead",
        "DE_load_forecast_entsoe_transparency",
        "DE_wind_generation_actual",
        "DE_solar_generation_actual"
    ]

    df = df[columns]

    df = df.rename(columns={
        "DE_LU_price_day_ahead": "price",
        "DE_load_forecast_entsoe_transparency": "load",
        "DE_wind_generation_actual": "wind",
        "DE_solar_generation_actual": "solar"
    })

    return df