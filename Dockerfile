# Dockerfile 예시
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 복사
COPY model.pkl .

# 모델을 사용할 수 있는 스크립트 추가
COPY predict.py .

CMD ["python", "predict.py"]
