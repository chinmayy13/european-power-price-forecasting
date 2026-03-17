import pandas as pd

def clean_data(df):

    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

    df = df.sort_values("utc_timestamp")

    df = df.dropna()
    
    print("Checking duplicate timestamps...")
    duplicates = df["utc_timestamp"].duplicated().sum()
    print("Duplicate timestamps:", duplicates)

    print("Checking missing values...")
    print(df.isnull().sum())

    return df