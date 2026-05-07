import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

files = glob.glob("data/*.csv")

df_list = []

for file in files:

    print(f"Reading: {file}")

    temp_df = pd.read_csv(
        file,
        low_memory=False,
        nrows=30000
    )

    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

print("Dataset loaded!")
print(df.shape)

# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()

# =========================
# CLEAN DATA
# =========================

print("Cleaning dataset...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df.drop_duplicates(inplace=True)

print("Cleaning completed!")

# =========================
# REMOVE RARE CLASSES
# =========================

print("Removing rare classes...")

label_counts = df['Label'].value_counts()

valid_labels = label_counts[label_counts > 10].index

df = df[df['Label'].isin(valid_labels)]

print(df['Label'].value_counts())

# =========================
# SAMPLE DATASET
# =========================

print("Sampling dataset...")

if len(df) > 50000:
    df = df.sample(50000, random_state=42)

print(df.shape)

# =========================
# SELECT FEATURES
# =========================

selected_features = [

    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'SYN Flag Count',
    'ACK Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'URG Flag Count'
]

X = df[selected_features]

y = df['Label']

# =========================
# LABEL ENCODING
# =========================

print("Encoding labels...")

le = LabelEncoder()

y = le.fit_transform(y)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

print("Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================
# APPLY SMOTE
# =========================

print("Applying SMOTE...")

smote = SMOTE(
    random_state=42,
    k_neighbors=1
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE completed!")

# =========================
# TRAIN MODEL
# =========================

print("Training SVM model...")

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

model.fit(X_train, y_train)

print("Training completed!")

# =========================
# PREDICTION
# =========================

print("Predicting...")

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/svm_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

joblib.dump(le, "models/label_encoder.pkl")

print("Model saved successfully!")