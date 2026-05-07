import pandas as pd
import numpy as np

# Tạo 79 cột giả lập như bộ dữ liệu gốc [cite: 31]
columns = [f'feature_{i}' for i in range(79)]
# Đổi tên một số cột quan trọng để làm Lab 2.4 [cite: 67-78]
important_cols = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]
for i, col in enumerate(important_cols):
    columns[i] = col

# Tạo dữ liệu ngẫu nhiên
data = np.random.rand(100, 79)
df = pd.DataFrame(data, columns=columns)
df['Label'] = ['BENIGN'] * 90 + ['DDoS'] * 10 # Giả lập class imbalance [cite: 33]

# Lưu thành file csv
df.to_csv('data/cicids2017.csv', index=False)
print("Đã tạo file dữ liệu mẫu thành công!")
