"""
File: data_cleaning.py
Description: Cleaning the framingham.csv dataset.
"""

import pandas as pd
import numpy as np

# original dataset
raw_data = "../data/framingham.csv"

# clean dataset out path
clean_data = "../data/framingham_cleaned.csv"

def clean_heart_data():
    # load raw data
    df = pd.read_csv(raw_data)

    # dropping 2 columns
    df = df.drop(columns=["education", "glucose"])

    # drop rows with missing values in remaining columns
    df_clean = df.dropna()

    # save cleaned dataset
    df_clean.to_csv(clean_data, index=False)

    print(f"Cleaned dataset saved to {clean_data}.")
    print(f"Rows before dropping: {len(df)}, rows after dropping: {len(df_clean)}")

if __name__ == "__main__":
    clean_heart_data()
