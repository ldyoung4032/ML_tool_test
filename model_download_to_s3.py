import boto3

s3 = boto3.client('s3')
bucket_name = 'my-mlflow-dvc-bucket'
s3_key = 'mlflow/1/models/m-a0cb34712c6043349fc3c8352a26f633/artifacts/model.pkl'
local_path = 'model_s3.pkl'

s3.download_file(bucket_name, s3_key, local_path)


import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# S3에서 이미 다운로드한 model_s3.pkl 파일을 로드합니다.
with open('model_s3.pkl', 'rb') as f:
    model = pickle.load(f)

# iris 데이터셋 불러오기
iris = load_iris()
X, y = iris.data, iris.target

# 학습/테스트 데이터 분할 (여기서는 예측 테스트용으로만 사용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 모델로 예측 수행
y_pred = model.predict(X_test)

# 결과 출력
print("예측 결과:", y_pred)
print("\n분류 리포트:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
