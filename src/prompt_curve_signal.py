import pandas as pd


def generate_signal(predicted_price, market_price, threshold=1.0):

    diff = predicted_price - market_price

    if diff > threshold:
        return "LONG"

    elif diff < -threshold:
        return "SHORT"

    else:
        return "HOLD"