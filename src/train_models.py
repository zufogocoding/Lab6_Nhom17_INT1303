import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import the 5 required models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC  # Thay đổi: Dùng LinearSVC thay vì SVC tốn RAM
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def load_processed_data(processed_data_dir):
    """Loads the preprocessed train and test datasets."""
    print("[1/4] Loading processed data...")
    train_path = os.path.join(processed_data_dir, "train_processed.csv")
    test_path = os.path.join(processed_data_dir, "test_processed.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Processed data not found. Please run data_preprocessing.py first."
        )

    # TỐI ƯU RAM 1: Trích xuất ngẫu nhiên 15% tập train để học, tránh tràn RAM
    train_df = pd.read_csv(train_path).sample(frac=0.15, random_state=42)

    # Tập test giữ nguyên 100% để đảm bảo kết quả đánh giá khách quan
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # Load Label Encoder to get class names for our charts
    le = joblib.load("../models/label_encoder.pkl")
    class_names = le.classes_

    return X_train, y_train, X_test, y_test, class_names


def evaluate_and_plot(y_test, y_pred, model_name, class_names):
    """Prints the classification report and plots a confusion matrix."""
    print(f"\n--- Evaluation Report: {model_name} ---")
    print(
        classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    )

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix: {model_name}\n(Pay attention to Attack Recall!)")

    # Save the plot to the visuals folder
    os.makedirs("../visuals", exist_ok=True)
    plt.savefig(f"../visuals/cm_{model_name.replace(' ', '_')}.png")
    plt.close()


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names):
    """Trains all 5 models, evaluates them, and returns the trained Random Forest."""
    print("[2/4] Initializing models...")

    # TỐI ƯU RAM 2: Cấu hình giới hạn vòng lặp (max_iter) và tận dụng đa luồng (n_jobs=-1)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200, n_jobs=-1, random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SGDClassifier(loss="hinge", random_state=42, n_jobs=-1, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
    }

    rf_trained_model = None

    print("[3/4] Training and evaluating models (this might take a few minutes)...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        print(f"Predicting with {name}...")
        y_pred = model.predict(X_test)

        # Evaluate
        evaluate_and_plot(y_test, y_pred, name, class_names)

        # Save the Random Forest model specifically for the next step
        if name == "Random Forest":
            rf_trained_model = model

    return rf_trained_model


def save_best_model(model, output_dir):
    """Saves the best model (Random Forest) to the models directory."""
    print("\n[4/4] Saving the Best Model (Random Forest)...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "random_forest_best_model.pkl")
    joblib.dump(model, model_path)
    print(f"Success! Model saved to: {model_path}")


if __name__ == "__main__":
    PROCESSED_DATA_DIR = "../data/processed"
    MODELS_DIR = "../models"

    try:
        # Pipeline execution
        X_train, y_train, X_test, y_test, class_names = load_processed_data(
            PROCESSED_DATA_DIR
        )

        # Train, evaluate, and extract the RF model
        best_model = train_and_evaluate(X_train, y_train, X_test, y_test, class_names)

        # Save it for deployment
        if best_model is not None:
            save_best_model(best_model, MODELS_DIR)

        print(
            "\nAll training complete. Check the '../visuals' folder for your confusion matrices!"
        )

    except Exception as e:
        print(f"An error occurred: {e}")
