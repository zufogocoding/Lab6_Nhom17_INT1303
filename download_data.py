import kagglehub
import shutil
import os

print("Đang tải dữ liệu từ Kaggle (có thể mất vài phút)...")
# Tải bộ dữ liệu về thư mục cache
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

# Đảm bảo thư mục data/raw đã tồn tại
os.makedirs("data/raw", exist_ok=True)

# Copy các file CSV từ thư mục cache sang thư mục dự án
count = 0
for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), os.path.join("data/raw", file))
        count += 1

print(f"Thành công! Đã chuyển {count} file CSV vào thư mục data/raw/")
