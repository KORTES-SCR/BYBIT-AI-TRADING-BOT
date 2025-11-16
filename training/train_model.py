# training/train_model.py

import pandas as pd
import numpy as np
import pandas_ta as ta # Make sure pandas_ta is installed
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LayerNormalization, Dense, Dropout, Add,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
# from binance.client import Client # No longer needed
import os
from dotenv import load_dotenv
import traceback

# Import our modules
from .data_loader import download_data
from .feature_engineering import generate_features

# --- SETTINGS ---
# Look for .env one level up (in project root)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path) 

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')
PAIR = os.getenv("PAIR", "SOLUSDT")
INTERVAL = '60' # Bybit interval '60' for 1H
DATA_POINTS = 30000 # ~1.1 years of hourly data
CACHE_DIR = "../data" # Data folder one level up
CACHE_FILE = os.path.join(CACHE_DIR, f"{PAIR}_{INTERVAL}_cache.parquet")

# --- Data labeling parameters ---
LOOK_FORWARD_CANDLES = 1 # Forecast 1 hour ahead

# --- Model parameters ---
SEQUENCE_LENGTH = 48 # Window size (48 hourly candles = 2 days)
# Paths for saving to project root (one level up)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_bybit_1h.keras') # New name
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_bybit_1h.pkl')
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_bybit_1h.parquet')

# --- Transformer/CNN parameters ---
NUM_HEADS = 4; FF_DIM = 64; NUM_TRANSFORMER_BLOCKS = 1
CNN_FILTERS = 32; CNN_KERNEL_SIZE = 3

# --- 1. DATA LOADING (via Bybit data_loader) ---
df_raw = download_data(PAIR, INTERVAL, DATA_POINTS, API_KEY, API_SECRET, USE_TESTNET, CACHE_FILE)
if df_raw.empty: print("❌ Failed to load data."); exit()

# --- 2. FEATURE GENERATION (via feature_engineering) ---
df = generate_features(df_raw)

# --- ❗️ TRUNCATED FEATURE LIST (without Taker Volume, N_Trades) ---
base_ohlcv = ['open', 'high', 'low', 'volume'] # 'close' is not a feature
returns = [f'return_{lag}h' for lag in [1, 3, 5, 15, 60]]
emas = [f'EMA_{length}' for length in [9, 21, 55, 100, 200]]
mas = [f'MA_{length}' for length in [50, 100]]
# Look for StochRSI columns created by feature_engineering
stoch_cols = [col for col in df.columns if col.startswith('STOCHRSI')]
momentum = ['RSI_14', 'RSI_28', 'MACD', 'MACD_h', 'MACD_s', 'ATR_14', 'MFI_14'] + stoch_cols
volatility = ['BBU', 'BBL', 'BBB', 'STD_20']
volume_f = ['volume_change', 'OBV'] # ❗️ Removed taker_buy_ratio
time_f = ['hour', 'weekday']
rolling_f = ['rolling_mean_6h', 'rolling_std_6h', 'rolling_std_48h',
             'rolling_skew_24h', 'rolling_skew_48h',
             'rolling_kurt_24h', 'rolling_kurt_48h']

feature_list = base_ohlcv + returns + emas + mas + momentum + volatility + volume_f + time_f + rolling_f
# Check if all columns from the list exist in DataFrame
actual_cols = [f for f in feature_list if f in df.columns]
missing_cols = list(set(feature_list) - set(actual_cols))
if missing_cols: 
    print(f"⚠️ WARNING: Failed to create/find columns: {missing_cols}. They will be excluded from training.")
    feature_list = actual_cols # Use only existing columns
print(f"\nFinal feature list ({len(feature_list)}): {feature_list}")

# --- 3. CREATING TARGET VARIABLE ---
print("Creating target variable (sign)...")
# 'future_real_return' not needed for training, only for backtest.py
df['future_close'] = df['close'].shift(-LOOK_FORWARD_CANDLES)
df['label'] = (df['future_close'] > df['close']).astype(int)
df.dropna(subset=['future_close'], inplace=True) # Remove LAST rows

# --- 4. Cleaning NaN in features ---
print(f"Rows before final NaN/inf cleaning: {len(df)}")
# Make sure feature_list contains only columns that exist in df
feature_list = [f for f in feature_list if f in df.columns] 
df_features_only = df[feature_list].copy()
df_features_only.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features_only.ffill(inplace=True); df_features_only.bfill(inplace=True)
remaining_nan_cols = df_features_only.isnull().sum(); remaining_nan_cols = remaining_nan_cols[remaining_nan_cols > 0]
if not remaining_nan_cols.empty: 
    print(f"❌ CRITICAL ERROR: NaN remained AFTER ffill+bfill in columns:")
    print(remaining_nan_cols)
    print("Training is impossible.")
    exit()
else: print("✅ NaN check after ffill+bfill: No gaps found.")
# Align main df with cleaned features (by index)
df = df.loc[df_features_only.index]

# --- 5. SAVING PREPARED DATA FOR BACKTESTER ---
print(f"Saving prepared data to {PROCESSED_DATA_FILE}...")
# Save ALL columns that may be needed by backtester
# (OHLCV + All features + label + future_close)
cols_to_save = ['open', 'high', 'low', 'close', 'volume'] + feature_list + ['label', 'future_close']
cols_to_save = sorted(list(set(cols_to_save))) # Remove duplicates and sort
df_to_save = df[cols_to_save].copy()
try:
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    df_to_save.to_parquet(PROCESSED_DATA_FILE)
    print("Data saved.")
except Exception as e_save: 
    print(f"❌ Parquet save error: {e_save}"); exit()


# --- 6. PREPARING DATA FOR MODEL TRAINING ---
print("Scaling data for training...")
scaler = StandardScaler()
# Train scaler and scale cleaned features
scaled_features = scaler.fit_transform(df_features_only)

print("Creating windows for training...")
y_data = df['label'].values # Take label from original df
X, y = [], []
if len(scaled_features) != len(y_data): print("❌ ERROR: X and y length mismatch!"); exit()
# Determine real window length
current_sequence_length = min(SEQUENCE_LENGTH, len(scaled_features) - 1)
if current_sequence_length < 10: print("❌ Too little data for windows!"); exit()
elif current_sequence_length != SEQUENCE_LENGTH: print(f"⚠️ Window length reduced to {current_sequence_length}.")

for i in range(current_sequence_length, len(scaled_features)):
    X.append(scaled_features[i-current_sequence_length:i])
    y.append(y_data[i])
if not X: print("❌ ERROR: Failed to create data windows!"); exit()
X, y = np.array(X), np.array(y)
print(f"Training data size: X={X.shape}, y={y.shape}")

# --- 7. BUILDING and TRAINING FINAL MODEL ---
print("Building CNN-Transformer model...")

def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    """Transformer Encoder Block"""
    x = LayerNormalization(epsilon=1e-6)(inputs)
    # key_dim must be positive
    key_dim = max(1, inputs.shape[-1] // num_heads) 
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = Dropout(dropout)(attention_output)
    x = Add()([inputs, attention_output]) # Residual connection 1

    x_ff = LayerNormalization(epsilon=1e-6)(x)
    x_ff = Dense(ff_dim, activation="relu")(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff]) # Residual connection 2
    return x

def build_cnn_transformer_model(input_shape, num_heads, ff_dim, cnn_filters, cnn_kernel_size, num_transformer_blocks=1, dropout=0.1):
    """Builds hybrid CNN + Transformer model"""
    inputs = Input(shape=input_shape)
    
    # --- CNN Block ---
    # padding='causal' ensures convolution doesn't "look into the future"
    x = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', padding='causal')(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Conv1D(filters=cnn_filters * 2, kernel_size=cnn_kernel_size, activation='relu', padding='causal')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # --- Transformer Block ---
    transformer_input = x # Pass CNN output to transformer
    
    for _ in range(num_transformer_blocks):
        transformer_input = transformer_encoder(transformer_input, num_heads, ff_dim, dropout)
    
    # --- Final layers ---
    x = GlobalAveragePooling1D()(transformer_input) # Aggregate transformer outputs
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x) # Sigmoid output for binary classification (0=SHORT, 1=LONG)
    
    return Model(inputs=inputs, outputs=outputs)

# Проверка формы X перед построением
input_shape_model = (current_sequence_length, len(feature_list))
if X.shape[1:] != input_shape_model: 
     print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Неожиданная форма данных X: {X.shape}. Ожидалось: ({X.shape[0]}, {input_shape_model[0]}, {input_shape_model[1]})")
     exit()

model = build_cnn_transformer_model(
    input_shape=input_shape_model, num_heads=NUM_HEADS, ff_dim=FF_DIM,
    cnn_filters=CNN_FILTERS, cnn_kernel_size=CNN_KERNEL_SIZE,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS
)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print(f"Обучение финальной модели на {X.shape[0]} примерах...")
# Обучаем на всех данных (X_final, y_final), без validation_split, для финальной модели
model.fit(X, y, epochs=30, batch_size=32, verbose=1) 

# --- 8. СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ И Скейлера ---
print("Сохранение финальной модели и скейлера...")
try:
    model.save(MODEL_FILE, save_format='keras')
    # Переобучаем и сохраняем scaler на ПОЛНОМ наборе фичей ДО создания окон
    scaler_final = StandardScaler().fit(df_features_only) # Обучаем на df_features_only
    joblib.dump(scaler_final, SCALER_FILE)
    print(f"Финальная модель сохранена как {MODEL_FILE}")
    print(f"Финальный скейлер сохранен как {SCALER_FILE}")
except Exception as e_save_model: 
    print(f"❌ Ошибка сохранения: {e_save_model}")
    traceback.print_exc(limit=2)