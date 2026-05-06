# ==========================================
# File: 1_data_preprocessing.py
# Chức năng:
# - Tải dataset CIC-IDS2017 từ Kaggle
# - Gộp 8 file CSV, làm sạch dữ liệu
# - Vẽ biểu đồ phân phối lớp & heatmap tương quan
# - Mã hóa nhãn, chia train/test
# - Xử lý mất cân bằng (SMOTE + RandomUnderSampler)
# - Scale dữ liệu, lưu ra file .npy
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ---------- 1. Tải dataset ----------
print("Đang tải dataset từ Kaggle...")
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
print(f"Tìm thấy {len(csv_files)} file CSV.")

# Gộp tất cả CSV
df_list = []
for file in csv_files:
    df_list.append(pd.read_csv(os.path.join(path, file)))
df = pd.concat(df_list, ignore_index=True)
print(f"Kích thước dữ liệu gốc: {df.shape}")

# ---------- 2. Làm sạch ----------
df.columns = df.columns.str.strip()  # Xóa khoảng trắng thừa
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Điền NaN bằng median từng cột số
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Xóa cột chỉ chứa 1 giá trị duy nhất (trừ cột Label)
zero_var = [c for c in df.columns if df[c].nunique() <= 1 and c != "Label"]
df.drop(columns=zero_var, inplace=True)

# Xóa hàng trùng lặp
df.drop_duplicates(inplace=True)

# Giảm bộ nhớ (downcast)
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

print(f"Kích thước sau làm sạch: {df.shape}")

# ---------- 3. EDA (lưu ảnh để xem) ----------
# Biểu đồ phân phối nhãn
plt.figure(figsize=(12, 5))
df["Label"].value_counts().plot(kind="bar")
plt.title("Phân phối loại tấn công")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("phan_phoi_nhan.png")
plt.close()

# 18 đặc trưng chính (theo yêu cầu pdf)
selected_features = [
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byt/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "URG Flag Cnt",
]
# Chỉ giữ đặc trưng có thật trong dataset
available_feats = [f for f in selected_features if f in df.columns]
print(f"Số đặc trưng được dùng: {len(available_feats)}")

# Heatmap tương quan
plt.figure(figsize=(14, 10))
corr = df[available_feats].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Heatmap tương quan giữa 18 đặc trưng")
plt.tight_layout()
plt.savefig("heatmap_tuong_quan.png")
plt.close()

# ---------- 4. Chuẩn bị X, y ----------
X = df[available_feats].copy()
y = df["Label"].copy()

# Mã hóa nhãn
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Lưu danh sách tên lớp để dùng sau
np.save("models/label_classes.npy", le.classes_)

# Chia train/test (80/20, giữ cân bằng lớp)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ---------- 5. Cân bằng dữ liệu ----------
print("Đang cân bằng dữ liệu (SMOTE + undersampling)...")
over = SMOTE(sampling_strategy=0.1, random_state=42)  # sinh mẫu thiểu số
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # giảm mẫu đa số
X_res, y_res = over.fit_resample(X_train, y_train)
X_res, y_res = under.fit_resample(X_res, y_res)

# Scale dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. Lưu trữ dữ liệu đã xử lý ----------
os.makedirs("models", exist_ok=True)
np.save("models/X_train_scaled.npy", X_train_scaled)
np.save("models/X_test_scaled.npy", X_test_scaled)
np.save("models/y_train.npy", y_res)
np.save("models/y_test.npy", y_test)
np.save("models/available_feats.npy", np.array(available_feats))

# Lưu scaler và encoder
import joblib

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Tiền xử lý hoàn tất! Dữ liệu đã lưu vào thư mục 'models/'")
