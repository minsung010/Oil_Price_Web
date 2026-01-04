# train_predict.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import chardet

# ----------------- 설정 -----------------
CSV_PATH = "../gas_price_lstm/data/주유소_지역별_평균판매가격.csv"   # 사용자 CSV 경로
SEQ_LEN = 50           # 입력 시퀀스 길이 (최근 50일로 예시)
Y_SIZE = 1             # 예측 길이 (다음 1일)
EPOCHS = 100
BATCH_SIZE = 16
PLOT_DIR = "static/plots"
MODEL_DIR = "models"
OUT_CSV = "prediction_summary.csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- 파일 인코딩 자동 감지 후 로드 -----------------
def detect_encoding(path, nbytes=4000):
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    return chardet.detect(raw)["encoding"]

enc = detect_encoding(CSV_PATH)
print("Detected file encoding:", enc)
df_raw = pd.read_csv(CSV_PATH, encoding=enc)

# ----------------- 데이터 형태(가로: 날짜 x 열: 지역) -> 세로(long) 변환 -----------------
# 예시 CSV 에서 첫 컬럼명이 '구분' 또는 'date' 계열이라 가정
# 날짜 컬럼 찾기
date_col = None
for c in df_raw.columns:
    if c.strip().lower() in ("구분", "date", "일자", "day"):
        date_col = c
        break
if date_col is None:
    # 첫 열을 날짜로 사용
    date_col = df_raw.columns[0]

# 날짜 컬럼을 표준 datetime 으로 변환 (형식: '2025년08월23일' 같은 경우도 처리)
def parse_korean_date(s):
    # '2025년08월23일' -> '2025-08-23'
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # 간단 통합 처리
    m = re.match(r"(\d{4})\D*(\d{1,2})\D*(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    try:
        return pd.to_datetime(s)
    except:
        return pd.NaT

df = df_raw.copy()
df[date_col] = df[date_col].apply(parse_korean_date)
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)
df = df.reset_index(drop=True)

# 나머지 컬럼들을 지역명으로 본다
region_cols = [c for c in df.columns if c != date_col]

# melt -> long format: date, region, price
df_long = pd.melt(df, id_vars=[date_col], value_vars=region_cols,
                  var_name="region", value_name="price")

df_long = df_long.rename(columns={date_col: "date"})
# numeric 변환
df_long['price'] = pd.to_numeric(df_long['price'], errors='coerce')
# 간단 보간: 지역별로 결측치를 선형 보간 후 앞뒤 채움
df_long['price'] = df_long.groupby('region')['price'].transform(lambda s: s.interpolate().ffill().bfill())


# ----------------- 시퀀스 생성 함수 -----------------
def create_sequences(series, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])  # 다음 값
    return np.array(X), np.array(y)

results = []
regions = df_long['region'].unique()
print("총 지역 수:", len(regions))

for region in regions:
    sub = df_long[df_long['region'] == region].sort_values('date').reset_index(drop=True)
    prices = sub['price'].values
    dates = sub['date'].values

    if len(prices) < SEQ_LEN + 2:
        print(f"[스킵] {region} — 데이터가 너무 적음 ({len(prices)} rows)")
        continue

    # 스케일링 (region 별)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1,1)).flatten()

    # 시퀀스 생성
    X, y = create_sequences(prices_scaled, seq_len=SEQ_LEN)
    # train/test split
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # reshape (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 모델 구성
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN,1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64),
        Dropout(0.2),
        Dense(Y_SIZE)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 학습
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    print(f"[학습] 지역: {region} / 데이터 길이: {len(prices)} / 학습샘플: {X_train.shape[0]} / 검증샘플: {X_test.shape[0]}")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

    # 마지막 SEQ로 다음날 예측
    last_seq = prices_scaled[-SEQ_LEN:].reshape((1, SEQ_LEN, 1))
    pred_scaled = model.predict(last_seq)
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]

    # 테스트 구간에 대한 예측 (시각화)
    pred_test_scaled = model.predict(X_test).flatten()
    pred_test = scaler.inverse_transform(pred_test_scaled.reshape(-1,1)).flatten()
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    # 시각화 저장 (파일명에 안전한 문자만)
    safe_region = re.sub(r'[\\/*?:"<>|]', "_", str(region))
    plot_path = os.path.join(PLOT_DIR, f"{safe_region}_predict.png")
    plt.figure(figsize=(12,6))
    plt.plot(range(len(y_test_unscaled)), y_test_unscaled, label='True (test)')
    plt.plot(range(len(pred_test)), pred_test, label='Predicted (test)')
    plt.title(f"{region} - True vs Predicted (test segment)")
    plt.xlabel("time index (test)")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # 모델 저장 (선택)
    try:
        model.save(os.path.join(MODEL_DIR, f"{safe_region}_model.keras"))
    except Exception as e:
        print("모델 저장 실패:", e)

    last_known_price = float(prices[-1])
    results.append({
        'region': str(region),
        'pred_price': float(pred_price),
        'plot_path': plot_path.replace("\\","/"),
        'last_known_price': last_known_price
    })

# 결과 CSV 저장
out_df = pd.DataFrame(results).sort_values('region')
out_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print("완료:", OUT_CSV, "및 plot 이미지들 저장됨 ->", PLOT_DIR)
