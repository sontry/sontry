import joblib
import numpy as np

# 모델 로드
model = joblib.load("model.pkl")

# 예측할 데이터 예시
data = np.array([[5.1, 3.5, 1.4, 0.2]])

# 예측 수행
prediction = model.predict(data)
print("Prediction:", prediction)
