import math
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file and handle encoding issues and missing markers.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        df.replace(["N/A", ""], np.nan, inplace=True)
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return pd.DataFrame()


def clean_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert 'Price' and 'Price/m²' columns to integers.
    """
    def parse(value):
        if pd.isna(value):
            return np.nan
        value = str(value).replace("\n", " ").replace(",", "").replace("EGP", "").replace("EGP/m", "")
        value = re.sub(r"[^\d]", "", value)
        return int(value) if value.isdigit() else np.nan

    df["Price"] = df["Price"].apply(parse)
    df["Price/m²"] = df["Price/m²"].apply(parse)
    return df


def clean_area_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric values from the 'Area' column.
    """
    def extract(value):
        if pd.isna(value):
            return np.nan
        value = str(value).replace("m²", "").replace("mÂ²", "").replace("Â²", "")
        match = re.search(r"\d+", value)
        return int(match.group()) if match else np.nan

    df["Area"] = df["Area"].apply(extract)
    return df


def convert_room_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert room counts to numeric.
    """
    df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors="coerce")
    df["Bathrooms"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values: mean for numeric, mode for categorical.
    """
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if col in ["Bedrooms", "Bathrooms"]:
                df[col] = df[col].fillna(math.ceil(df[col].mean()))
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def clean_location_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'Greater Cairo /' prefix from the Location column.
    """
    df["Location"] = df["Location"].str.replace(r"^Greater Cairo\s*/\s*", "", regex=True)
    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize continuous numerical features only.
    """
    scaler = StandardScaler()
    numeric_cols = ["Price", "Price/m²", "Area"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess(filepath: str) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline.
    """
    df = load_data(filepath)
    if df.empty:
        return df

    df = clean_price_columns(df)
    df = clean_area_column(df)
    df = convert_room_columns(df)
    df = clean_location_column(df)
    df = fill_missing_values(df)

    # Standardize numerical features
    # df = scale_numerical_features(df)

    # Clean index and unnamed columns
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=[col for col in df.columns if col.startswith("Unnamed:")], inplace=True)

    print("[INFO] Preprocessing complete.")
    return df
