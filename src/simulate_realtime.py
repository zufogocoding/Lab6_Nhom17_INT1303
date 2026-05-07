import joblib
import pandas as pd
import time
import random


def simulate_network_flow():
    print("Khởi động hệ thống phát hiện xâm nhập thời gian thực...")

    # Load model, label encoder, scaler
    try:
        model = joblib.load("../models/random_forest_best_model.pkl")
        le = joblib.load("../models/label_encoder.pkl")
    except FileNotFoundError:
        print(
            "Lỗi: Không tìm thấy file model hoặc encoder. Hãy chạy train_models.py trước."
        )
        return

    # Load file test đã xử lý để làm nguồn dữ liệu mô phỏng
    try:
        test_df = pd.read_csv("../data/processed/test_processed.csv")
        X_test = test_df.drop(columns=["Label"])
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy dữ liệu test.")
        return

    print("Đang giám sát mạng (Nhấn Ctrl+C để dừng)...\n")

    try:
        # Chạy vòng lặp vô hạn mô phỏng gói tin đến (giới hạn 100 lần thử cho an toàn)
        for _ in range(100):
            # Lấy ngẫu nhiên 1 gói tin từ tập test
            idx = random.randint(0, len(X_test) - 1)
            incoming_flow = X_test.iloc[[idx]]

            # Dự đoán
            prediction_encoded = model.predict(incoming_flow)[0]
            prediction_label = le.inverse_transform([prediction_encoded])[0]

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Kiểm tra và in cảnh báo
            if prediction_label != "BENIGN":
                # Do các feature đã chọn không có Port, giả lập Destination Port = 80 cho đúng định dạng Suricata
                alert = f"[{timestamp}] [ALERT] Suspicious traffic detected: {prediction_label}. Destination Port: 80."
                print(alert)
                # Ghi log ra file
                with open("../alerts.log", "a") as f:
                    f.write(alert + "\n")
            else:
                print(f"[{timestamp}] [INFO] Normal traffic flow.")

            time.sleep(0.5)  # Dừng 0.5s giữa các gói tin

    except KeyboardInterrupt:
        print("\nĐã dừng hệ thống giám sát.")


if __name__ == "__main__":
    simulate_network_flow()
