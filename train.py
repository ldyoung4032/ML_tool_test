import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

def load_data_from_package():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    print("[패키지 데이터셋 로드 완료]")
    return X, y

def save_split_data(X_train, X_test, y_train, y_test, out_dir="data/split"):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    print(f"[분할 데이터 저장 완료] → {out_dir}")

def train_and_log_model(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("random_forest_iris")
    print("[MLflow 실험 기록 시작]")
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
        joblib.dump(clf, "model.joblib")
        print(f"accuracy: {acc:.4f}")
        print("[모델 저장 완료: model.joblib, MLflow]")

def load_model(model_path="model.joblib"):
    clf = joblib.load(model_path)
    print("[모델 로드 완료]")
    return clf

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    X, y = load_data_from_package()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("[데이터 분할 완료]")
    save_split_data(X_train, X_test, y_train, y_test)
    train_and_log_model(X_train, X_test, y_train, y_test)
    # 모델 로드 예시
    model = load_model()
