# ==========================================
# File: 2_train_models.py
# Chức năng:
# - Train Logistic Regression, SVM, Naive Bayes, KNN, Random Forest
# - In báo cáo phân loại và vẽ confusion matrix
# - Lưu mô hình tốt nhất (Random Forest) và Logistic Regression
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------- Load dữ liệu đã xử lý ----------
X_train = np.load("models/X_train_scaled.npy")
X_test = np.load("models/X_test_scaled.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

print("Kích thước train:", X_train.shape, "test:", X_test.shape)

# ---------- Định nghĩa các model ----------
models = {
    "Logistic Regression": LogisticRegression(
        multi_class="ovr",
        solver="saga",
        penalty="l2",
        max_iter=500,
        n_jobs=-1,
        random_state=42,
    ),
    "SVM": SVC(kernel="rbf", random_state=42, probability=False),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}

results = {}
best_model_name = None
best_model = None
best_acc = 0

# ---------- Train & đánh giá ----------
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # Lưu kết quả vào bảng
    results[name] = {"accuracy": acc, "model": model, "y_pred": y_pred}

    # Vẽ confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=label_classes, yticklabels=label_classes
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    # Xác định model tốt nhất (dùng accuracy làm tiêu chí, có thể đổi sang F1)
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\n>>> Model tốt nhất: {best_model_name} (Accuracy {best_acc:.4f})")

# ---------- Lưu model tốt nhất và model phụ ----------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")  # Random Forest
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")

# Tạo bảng so sánh dạng CSV
summary = pd.DataFrame(results).T
summary.to_csv("model_comparison.csv")
print("Đã lưu bảng so sánh vào model_comparison.csv")
