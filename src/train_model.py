from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def train_model(df):

    features = [
"net_load",
"price_lag_1",
"price_lag_24",
"rolling_mean_24",
"hour",
"wind_lag_1",
"solar_lag_1",
"load_lag_1",
"net_load_lag_1",
"day_of_week"
]

    X = df[features]
    y = df["price"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # BASELINE MODEL (persistence)

    baseline_pred = y_test.shift(1).dropna()
    baseline_actual = y_test[1:]

    baseline_mae = mean_absolute_error(baseline_actual, baseline_pred)

    print(f"Baseline MAE (persistence): {baseline_mae:.2f} €/MWh")

    #XGBoost MODEL

    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test