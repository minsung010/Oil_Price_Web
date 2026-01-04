# train_flask_model.py
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import time

# ----------------- GPU 설정 (GPU Configuration) -----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용 가능 (GPU available): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU 미탐지 (No GPU detected) — CPU로 실행 중 (Using CPU, slower)")

# ----------------- 경로 설정 (Path Settings) -----------------
DATA_DIR = "monthly_csvs"
SEQ_LEN = 60
EPOCHS = 30
BATCH_SIZE = 128
MODEL_PATH = "models/global_lstm.keras"
SCALER_PATH = "scalers/global_scaler.save"
ENCODER_PATH = "scalers/uni_encoder.save"
PLOT_PATH = "static/plots/global_model_loss.png"
OUT_CSV = "prediction_summary.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# ----------------- CSV 병합 (Merge CSV files) -----------------
file_list = glob(os.path.join(DATA_DIR, "*.csv"))
all_data = []
for f in file_list:
    try:
        df = pd.read_csv(f, encoding='cp949')
        all_data.append(df)
    except Exception as e:
        print(f"[경고 Warning] {f} 읽기 실패 (Failed to read):", e)

data = pd.concat(all_data, ignore_index=True)
data = data.rename(columns={"번호": "UNI_ID", "기간": "DATE", "휘발유": "B027", "경유": "D047"})
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')
data = data.sort_values(['UNI_ID', 'DATE'])
data[['B027', 'D047']] = data.groupby('UNI_ID')[['B027', 'D047']].ffill()

print(f"✅ 데이터 로드 완료 (Data loaded) — {len(data['UNI_ID'].unique())}개 주유소 (stations)")

# ----------------- 인코딩 & 스케일링 (Encoding & Scaling) -----------------
encoder = LabelEncoder()
data['UNI_ENC'] = encoder.fit_transform(data['UNI_ID'])
scaler = MinMaxScaler()
data['B027_scaled'] = scaler.fit_transform(data[['B027']])

joblib.dump(scaler, SCALER_PATH)
joblib.dump(encoder, ENCODER_PATH)

# ----------------- 시퀀스 생성 (Create sequences for LSTM) -----------------
def create_global_sequences(df, seq_len=SEQ_LEN):
    X_seq, X_id, y = [], [], []
    for uid in df['UNI_ENC'].unique():
        sub = df[df['UNI_ENC'] == uid]
        prices = sub['B027_scaled'].values
        for i in range(len(prices) - seq_len):
            X_seq.append(prices[i:i + seq_len])
            X_id.append(uid)
            y.append(prices[i + seq_len])
    return np.array(X_seq), np.array(X_id), np.array(y)

X_seq, X_id, y = create_global_sequences(data)
print(f"✅ 시퀀스 생성 완료 (Sequence generation done): {len(X_seq):,}개 샘플 (samples)")

# ----------------- 학습/검증 분리 (Train/Validation split) -----------------
split_idx = int(len(X_seq) * 0.9)
X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
X_id_train, X_id_val = X_id[:split_idx], X_id[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], SEQ_LEN, 1))
X_seq_val = X_seq_val.reshape((X_seq_val.shape[0], SEQ_LEN, 1))

# ----------------- 학습 진행률 콜백 (Progress + ETA) -----------------
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        avg_time = np.mean(self.epoch_times)
        remaining = avg_time * (self.params['epochs'] - (epoch + 1))
        print(f"⏱ Epoch {epoch+1}/{self.params['epochs']} 완료 — 걸린 시간: {elapsed:.1f}s, 예상 남은 시간: {remaining/60:.1f}분")

time_callback = TimeHistory()

# ----------------- 모델 불러오기 또는 새로 생성 (Load or Build Model) -----------------
if os.path.exists(MODEL_PATH):
    print("✅ 기존 모델 불러오기 (Resuming training from saved model)")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🆕 새로운 모델 학습 시작 (Starting fresh training)")
    id_input = Input(shape=(1,), name="station_id")
    id_embed = Embedding(input_dim=len(encoder.classes_), output_dim=8)(id_input)
    id_flat = Flatten()(id_embed)
    id_dense = Dense(8, activation='relu')(id_flat)

    seq_input = Input(shape=(SEQ_LEN, 1), name="price_sequence")
    x = LSTM(128, return_sequences=True)(seq_input)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)

    merged = Concatenate()([x, id_dense])
    out = Dense(1)(merged)

    model = Model(inputs=[seq_input, id_input], outputs=out)
    model.compile(optimizer='adam', loss='mse')

# ----------------- 콜백 설정 (Callbacks) -----------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

# ----------------- 모델 학습 (Train the Model) -----------------
history = model.fit(
    {"price_sequence": X_seq_train, "station_id": X_id_train},
    y_train,
    validation_data=({"price_sequence": X_seq_val, "station_id": X_id_val}, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, time_callback],
    verbose=1
)

# ----------------- 성능 평가 (Performance Evaluation) -----------------
y_pred = model.predict({"price_sequence": X_seq_val, "station_id": X_id_val}, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\n📈 모델 성능 평가 (Model Performance Evaluation)")
print(f"  - RMSE: {rmse:.6f}")
print(f"  - MAE : {mae:.6f}")
print(f"  - R²  : {r2:.6f}")

# ----------------- 한글 폰트 설정 (Fix Korean Font Warning) -----------------
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------- 손실 그래프 저장 (Save Loss Graph) -----------------
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss (훈련 손실)')
plt.plot(history.history['val_loss'], label='Validation Loss (검증 손실)')
plt.legend()
plt.title("Global LSTM Loss — 손실 그래프")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

# ----------------- 7일 예측 (7-day Forecast per Station) -----------------
results = []
for uni_id in data['UNI_ID'].unique():
    sub = data[data['UNI_ID'] == uni_id].sort_values('DATE')
    enc_id = encoder.transform([uni_id])[0]
    seq = sub['B027_scaled'].values[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    preds_scaled = []
    for _ in range(7):
        next_val = model.predict({"price_sequence": seq, "station_id": np.array([[enc_id]])}, verbose=0)[0, 0]
        preds_scaled.append(next_val)
        seq = np.append(seq[:, 1:, :], [[[next_val]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    results.append({
        "UNI_ID": uni_id,
        "pred_day7_price": float(preds[-1]),
        "avg_predicted_price_7days": float(np.mean(preds))
    })

pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n✅ 예측 완료: {len(results)}개 주유소")
print(f"✅ 결과 CSV: {OUT_CSV}")
print(f"✅ 손실 그래프: {PLOT_PATH}")
print(f"✅ 모델 저장됨: {MODEL_PATH}")
