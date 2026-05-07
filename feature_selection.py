import pandas as pd
import numpy as np

# 1. Tải dữ liệu mẫu vừa tạo (Phần 2.2)
print("Đang đọc dữ liệu...")
df = pd.read_csv('data/cicids2017.csv')

# 2. Làm sạch tên cột (Phần 2.2) [cite: 52]
# Loại bỏ khoảng trắng thừa ở đầu/cuối tên cột để tránh lỗi KeyError
df.columns = df.columns.str.strip()

# 3. Định nghĩa 18 tính năng cốt lõi (Phần 2.4) [cite: 67-78]
selected_features = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

# 4. Thực hiện lọc dữ liệu [cite: 66]
# Giữ lại 18 cột tính năng và cột 'Label' (nhãn loại tấn công)
X = df[selected_features]
y = df['Label']

print("--- Kết quả lọc tính năng ---")
print(f"Số lượng tính năng ban đầu: {df.shape[1] - 1}") # Trừ cột Label
print(f"Số lượng tính năng sau khi lọc: {X.shape[1]}")
print(f"Danh sách các cột đã chọn: \n{X.columns.tolist()}")

# 5. Lưu dữ liệu đã lọc ra file mới (Phần 3: Deliverables) [cite: 101]
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv('data/processed_data.csv', index=False)
print("\nĐã lưu dữ liệu sạch vào: data/processed_data.csv")
