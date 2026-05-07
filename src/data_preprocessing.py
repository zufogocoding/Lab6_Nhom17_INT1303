import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
from pandas.api.types import is_numeric_dtype  # Dòng import vừa được thêm

# Define the 18 core features required by the lab
SELECTED_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "ACK Flag Cnt",
    "Pkt Len Std",
    "SYN Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "URG Flag Cnt",
]


def load_and_merge_data(raw_data_dir):
    """Loads all CSVs in the directory and merges them into one DataFrame."""
    print("[1/6] Loading and merging raw CSV files...")
    all_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(merged_df)}")
    return merged_df


def clean_data(df):
    """Strips columns, forces numeric types, handles NaN/Inf, and drops duplicates & zero-variance."""
    print("[2/6] Cleaning data...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Ép kiểu tất cả các cột (trừ Label) về dạng số. Các giá trị string lỗi sẽ bị biến thành NaN
    cols_to_numeric = [col for col in df.columns if col != "Label"]
    df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

    # Replace Infinity with NaN, then fill NaN with the median of each column
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Drop zero-variance features (columns where all values are the same)
    variances = df[numeric_cols].var()
    zero_var_cols = variances[variances == 0].index
    df.drop(columns=zero_var_cols, inplace=True)

    return df


def reduce_mem_usage(df):
    """Downcasts data types to save RAM."""
    print("[3/6] Optimizing memory usage...")
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()

            if pd.isna(c_min) or pd.isna(c_max):
                continue

            col_type = str(df[col].dtype)

            if "int" in col_type:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif "float" in col_type:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory decreased from {start_mem:.2f} MB to {end_mem:.2f} MB.")
    return df


def filter_features(df):
    """Filters dataset to keep only the 18 selected features and the Label."""
    print("[4/6] Filtering features...")
    features_to_keep = SELECTED_FEATURES + ["Label"]
    missing_cols = [col for col in features_to_keep if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    return df[features_to_keep]


def preprocess_and_balance(df, output_dir):
    """Encodes, splits, scales, and balances the training data."""
    print("[5/6] Encoding, scaling, and handling class imbalance...")

    # Encode Label
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    os.makedirs("../models", exist_ok=True)
    joblib.dump(le, "../models/label_encoder.pkl")

    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "../models/scaler.pkl")

    print("Applying SMOTE and RandomUnderSampler to Training Data...")
    resample_pipeline = Pipeline(
        [
            ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
            ("under", RandomUnderSampler(sampling_strategy="auto", random_state=42)),
        ]
    )

    X_train_balanced, y_train_balanced = resample_pipeline.fit_resample(
        X_train_scaled, y_train
    )

    print("[6/6] Saving processed datasets...")
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train_balanced, columns=SELECTED_FEATURES)
    train_df["Label"] = y_train_balanced
    train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=SELECTED_FEATURES)
    test_df["Label"] = y_test.values
    test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    RAW_DATA_DIR = "../data/raw"
    PROCESSED_DATA_DIR = "../data/processed"

    try:
        raw_df = load_and_merge_data(RAW_DATA_DIR)
        clean_df = clean_data(raw_df)
        optimized_df = reduce_mem_usage(clean_df)
        filtered_df = filter_features(optimized_df)
        preprocess_and_balance(filtered_df, PROCESSED_DATA_DIR)
    except Exception as e:
        print(f"An error occurred: {e}")
