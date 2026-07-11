import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Limit CPU threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def main():
    # ==========================
    # Load Dataset
    # ==========================
    df = pd.read_csv("data/churn.csv")

    print("Dataset Loaded Successfully")
    print(df.head())

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Remove missing values
    df = df.dropna()

    # ==========================
    # Encode Categorical Columns
    # ==========================
    encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    print("\nData After Encoding")
    print(df.head())

    # ==========================
    # Split Features & Target
    # ==========================
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ==========================
    # MLflow Experiment
    # ==========================
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run():

        # Train Model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nAccuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        # Log Parameters
        mlflow.log_param("Model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # Log Metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # ==========================
        # Log Model to MLflow
        # ==========================
        mlflow.sklearn.log_model(
            sk_model=model,
            name="Customer_Churn_Model"
        )

        # Save Model Locally
        joblib.dump(model, "models/churn_model.joblib")

    print("\nModel Saved Successfully")
    print("Model Logged to MLflow Successfully")


if __name__ == "__main__":
    main()