# Databricks notebook source
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

# COMMAND ----------

# MLflow 설정
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# COMMAND ----------

# 예측 및 정확도 계산
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# COMMAND ----------

# 모델 서명 추론
signature = infer_signature(X_train, model.predict(X_train))


# COMMAND ----------

# MLflow 로그
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model", signature=signature)
    joblib.dump(model, "model.pkl")
    
    print(f'Accuracy: {accuracy * 100:.2f}%')

