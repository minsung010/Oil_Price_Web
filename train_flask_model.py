import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 경로
DATA_PATH = "merged_fuel_data.csv"
MODEL_PATH = "models/global_lstm.keras"
SCALER_PATH = "scalers/global_scaler.save"
ENCODER_PATH = "scalers/uni_encoder.save"
SEQ_LEN = 60
OUT_CSV = "prediction_3days_fixed.csv"

# 데이터
data = pd.read_csv(DATA_PATH)
data['DATE'] = pd.to_datetime(data['DATE'])
uni_ids = data['UNI_ID'].unique()

# 스케일러/인코더
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# 모델
model = load_model(MODEL_PATH)

SEQ_LIST, ID_LIST, VALID_IDS = [], [], []

for uni_id in uni_ids:
    sub = data[data['UNI_ID'] == uni_id].sort_values('DATE')
    if len(sub) < SEQ_LEN:
        continue
    seq = sub['B027'].values[-SEQ_LEN:].reshape(-1, 1)
    seq_scaled = scaler.transform(seq)
    enc_id = encoder.transform([uni_id])[0]

    SEQ_LIST.append(seq_scaled)
    ID_LIST.append(enc_id)
    VALID_IDS.append(uni_id)

SEQ_ARRAY = np.array(SEQ_LIST)
ID_ARRAY = np.array(ID_LIST).reshape(-1, 1)

# 3일 예측
pred_days = 3
preds_all = np.zeros((len(VALID_IDS), pred_days), dtype=np.float32)
current_seq = SEQ_ARRAY.copy()

for day in range(pred_days):
    preds_scaled = model.predict(
        {"price_sequence": current_seq, "station_id": ID_ARRAY},
        batch_size=128,
        verbose=0
    )
    # 다음 시퀀스 업데이트
    current_seq = np.concatenate(
        [current_seq[:, 1:, :], preds_scaled.reshape(-1, 1, 1)],
        axis=1
    )
    preds_all[:, day] = preds_scaled.flatten()

# 역변환
preds_all_inversed = scaler.inverse_transform(preds_all.reshape(-1, 1)).reshape(len(VALID_IDS), pred_days)

# 저장
results = []
for i, uid in enumerate(VALID_IDS):
    results.append({
        "UNI_ID": uid,
        "pred_day1": float(preds_all_inversed[i, 0]),
        "pred_day2": float(preds_all_inversed[i, 1]),
        "pred_day3": float(preds_all_inversed[i, 2]),
        "avg_predicted_price_3days": float(np.mean(preds_all_inversed[i]))
    })

pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"✅ 3일 예측 완료, 저장: {OUT_CSV}")
