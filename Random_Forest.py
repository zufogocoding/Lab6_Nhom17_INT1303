import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Giả sử bạn đã có X_train_bal, y_train_bal (tập huấn luyện đã cân bằng) 
# và X_test_scaled, y_test (tập kiểm thử) từ bước tiền xử lý trước đó.

# 1. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
print("Đang khởi tạo và huấn luyện mô hình Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,      # Số lượng cây quyết định trong rừng. Càng nhiều cây mô hình càng ổn định nhưng chạy lâu hơn.
    max_depth=None,        # Độ sâu tối đa của cây. None nghĩa là cây mở rộng cho đến khi các lá hoàn toàn thuần nhất.
    random_state=42,       # Cố định random seed để kết quả chạy lại các lần đều giống nhau.
    n_jobs=-1,             # Sử dụng toàn bộ số nhân (cores) của CPU để huấn luyện nhanh hơn.
    class_weight='balanced' # Tự động điều chỉnh trọng số cho các lớp (giúp ích thêm nếu dữ liệu vẫn còn hơi lệch).
)

# Tiến hành huấn luyện
rf_model.fit(X_train_bal, y_train_bal)
print("Huấn luyện hoàn tất!\n")

# 2. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH

# Dự đoán trên tập test
y_pred_rf = rf_model.predict(X_test_scaled)

# In các chỉ số đánh giá cơ bản
print("--- KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST ---")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nBáo cáo phân loại chi tiết (Precision, Recall, F1-score):")
# Lưu ý: target_names có thể thay bằng le.classes_ nếu bạn dùng LabelEncoder ở bước trước
print(classification_report(y_test, y_pred_rf))

# Vẽ Confusion Matrix (Ma trận nhầm lẫn) để xem mô hình nhận diện sai ở đâu
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Nhãn dự đoán (Predicted)')
plt.ylabel('Nhãn thực tế (True)')
plt.show()

# ==========================================
# 3. MỨC ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG
# ==========================================
# Đoạn code này giúp bạn biết feature nào (VD: Flow Duration, Tot Fwd Pkts...) 
# đóng góp nhiều nhất vào quyết định của mô hình.
print("\nĐang tính toán độ quan trọng của các đặc trưng...")

# Giả sử 'selected_features' là list chứa tên 18 cột đặc trưng mạng bạn đã chọn
# Nếu chưa định nghĩa, thay thế bằng: features_names = [f'Feature {i}' for i in range(X_train_bal.shape[1])]
importances = rf_model.feature_importances_

# Sắp xếp độ quan trọng giảm dần
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Mức độ quan trọng của các đặc trưng (Feature Importances)")
plt.bar(range(X_train_bal.shape[1]), importances[indices], align="center", color='skyblue', edgecolor='black')
# plt.xticks(range(X_train_bal.shape[1]), [selected_features[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X_train_bal.shape[1]])
plt.tight_layout()
plt.show()