import pandas as pd

def create_features(df):


    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])

    df["hour"] = df["utc_timestamp"].dt.hour
    df["day_of_week"] = df["utc_timestamp"].dt.dayofweek
    df["month"] = df["utc_timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_24"] = df["price"].shift(24)
    df["price_lag_168"] = df["price"].shift(168)

    df["rolling_mean_24"] = df["price"].rolling(24).mean()

    df["price_volatility_24"] = df["price"].rolling(24).std()
    df["price_volatility_72"] = df["price"].rolling(72).std()

    df["net_load"] = df["load"] - (df["wind"] + df["solar"])

    df["wind_lag_1"] = df["wind"].shift(1)
    df["solar_lag_1"] = df["solar"].shift(1)
    df["load_lag_1"] = df["load"].shift(1)
    df["net_load_lag_1"] = df["net_load"].shift(1)

    df = df.dropna()

    return df