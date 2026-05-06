# ==========================================
# File: 3_simulate_realtime.py
# Chức năng:
# - Mô phỏng nhận luồng mạng thời gian thực
# - Dùng mô hình tốt nhất (Random Forest) để phân loại
# - Nếu không phải BENIGN, in ra cảnh báo dạng Suricata
# ==========================================

import numpy as np
import pandas as pd
import joblib
import random
import time

# Load model, scaler, label encoder, tên đặc trưng
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")
features = np.load("models/available_feats.npy", allow_pickle=True)

print("Mô phỏng phát hiện xâm nhập thời gian thực...")
print("(Nhấn Ctrl+C để dừng)\n")

# Sinh ngẫu nhiên một dòng dữ liệu mạng (minh họa)
# Trong thực tế bạn sẽ fetch từ card mạng, ở đây giả lập bằng dữ liệu test
X_test = np.load("models/X_test_scaled.npy")  # dùng chính tập test cũ để minh họa
y_test = np.load("models/y_test.npy")

# Giả lập gói tin đến mỗi 1 giây
for i in range(min(100, len(X_test))):  # chỉ thử 100 gói đầu
    # Lấy một mẫu ngẫu nhiên từ tập test
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test[idx].reshape(1, -1)

    # Dự đoán
    pred = model.predict(sample)[0]
    pred_label = le.inverse_transform([pred])[0]

    # In log nếu phát hiện tấn công
    if pred_label != "BENIGN":
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        alert = f"[{timestamp}] ALERT: Detected {pred_label} attack!"
        print(alert)
        # Ghi ra file log
        with open("alerts.log", "a") as f:
            f.write(alert + "\n")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normal traffic (BENIGN)")

    time.sleep(0.5)  # tạm dừng 0.5s để mô phỏng thời gian thực

print("Mô phỏng kết thúc.")
