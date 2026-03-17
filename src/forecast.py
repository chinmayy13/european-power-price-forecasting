import pandas as pd

def forecast_price(model, X_test):

    preds = model.predict(X_test)
    
    results = X_test.copy()
    results["forecast_price"] = preds

    return preds

