# ⛽ Gas Price Prediction & Service Web (주유소 가격 예측 서비스)

LSTM 딥러닝 모델을 활용하여 전국 주유소의 휘발유/경유 가격을 예측하고, Opinet API를 통해 실시간 최저가 주유소 정보를 제공하는 웹 서비스입니다.

## 📌 주요 기능 (Features)

*   **실시간 주유소 검색**: 지역별(시/군/구) 및 브랜드별 최저가 주유소 검색 (Opinet API 연동)
*   **가격 예측 (AI)**: LSTM 모델을 기반으로 향후 3일간의 유가 변동 추이 예측
*   **데이터 시각화**: 과거 가격 추이와 예측 가격을 그래프로 비교 분석
*   **상세 정보 제공**: 주유소별 부대시설(세차장, 편의점 등) 및 위치 정보 제공

## 🛠 기술 스택 (Tech Stack)

*   **Backend**: Python, Flask
*   **Frontend**: HTML, CSS, JavaScript (Template Engine)
*   **Database**: Oracle Database (cx_Oracle)
*   **AI/ML**: TensorFlow (Keras), Scikit-learn, NumPy, Pandas
*   **API**: Opinet 유가정보 API

## 📂 프로젝트 구조 (Project Structure)

```
gas_price_lstm/
├── app.py                 # Flask 메인 애플리케이션
├── train_predict.py       # LSTM 모델 학습 및 예측 스크립트
├── requirements.txt       # 의존성 패키지 목록
├── static/                # 정적 파일 (CSS, JS, Plot images)
├── templates/             # HTML 템플릿
├── models/                # 학습된 LSTM 모델 저장소 (.keras)
├── data/                  # 학습용 데이터셋 (CSV)
└── merged_fuel_data.csv   # 통합 연료 데이터
```

## 🚀 설치 및 실행 (Installation & Usage)

### 1. 환경 설정
Python 3.8 이상이 필요합니다.

```bash
# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터베이스 설정
Oracle DB가 로컬에 설치되어 있어야 하며, `app.py` 내의 DB 접속 정보를 본인 환경에 맞게 수정해야 할 수 있습니다.

```python
# app.py 설정 예시
DB_USER = "oil"
DB_PASSWORD = "oil"
DB_DSN = "localhost:1521/XE"
```

### 3. 애플리케이션 실행

```bash
python app.py
```
서버가 시작되면 브라우저에서 `http://localhost:5000` 으로 접속하세요.

## 📊 모델 학습 (Model Training)
새로운 데이터를 기반으로 모델을 재학습시키려면 아래 명령어를 실행하세요.

```bash
python train_predict.py
```
학습 결과는 `models/` 디렉토리에 저장되며, 예측 그래프는 `static/plots/`에 생성됩니다.

---
**Note**: 이 프로젝트는 학습 및 포트폴리오 목적으로 제작되었습니다.
