import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Tải dữ liệu đã qua lọc tính năng từ bước 2.4
df = pd.read_csv('data/processed_data.csv')

# 2. Chia dữ liệu thành Features (X) và Label (y)
X = df.drop('Label', axis=1)
y = df['Label']

# 3. Chia tập Train và Test (ví dụ 80% học, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện Random Forest [cite: 86]
print("Đang huấn luyện mô hình Random Forest... Vui lòng đợi.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Dự đoán và Đánh giá 
y_pred = model.predict(X_test)

print("\n--- BÁO CÁO PHÂN LOẠI (Classification Report) ---")
print(classification_report(y_test, y_pred))

# 6. Lưu mô hình để sử dụng cho Real-time (Mục 3: Deliverables) 
joblib.dump(model, 'random_forest_model.pkl')
print("\nĐã lưu mô hình thành công: random_forest_model.pkl")
