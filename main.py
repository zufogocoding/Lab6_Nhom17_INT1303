import pandas as pd

# Giả sử bạn có file dữ liệu (thay tên file tương ứng của bạn)
# df = pd.read_csv('data.csv')

# 1. Làm sạch tên cột (Phần 2.2) [cite: 52]
# df.columns = df.columns.str.strip()

# 2. Định nghĩa 18 tính năng quan trọng (Phần 2.4) [cite: 67-78]
selected_features = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

# 3. Lọc dữ liệu [cite: 66]
# X_selected = df[selected_features]

print("Đã thiết lập danh sách 18 tính năng thành công!")
