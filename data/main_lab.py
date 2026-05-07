import pandas as pd

# 1. Load dữ liệu [cite: 50]
df = pd.read_csv('data/cicids2017.csv')

# 2. Làm sạch tên cột (Tránh lỗi khoảng trắng thừa) [cite: 52]
df.columns = df.columns.str.strip()

# 3. Định nghĩa danh sách 18 tính năng cốt lõi [cite: 67-78]
selected_features = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

# 4. Thực hiện lọc (Feature Selection) 
# Giữ lại 18 tính năng + cột Label để huấn luyện
X_selected = df[selected_features]
y = df['Label']

print("--- KẾT QUẢ LAB 2.4 ---")
print(f"Số lượng cột ban đầu: {df.shape[1]}")
print(f"Số lượng cột sau khi lọc: {X_selected.shape[1]}")
print("Các cột được chọn:", X_selected.columns.tolist())

# Lưu lại kết quả để dùng cho phần 2.5 (Model Implementation) [cite: 79]
X_selected.to_csv('data/X_train_selected.csv', index=False)
