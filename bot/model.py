# bot/model.py
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import traceback

# --- SETTINGS ---
SEQUENCE_LENGTH = 48 # Should match train_model.py
MODEL_FILE = '../trading_model_bybit_1h.keras' # Path from config
SCALER_FILE = '../scaler_bybit_1h.pkl' # Path from config

# --- GLOBAL VARIABLES ---
MODEL = None
SCALER = None

# --- MODEL LOADING ---
try:
    MODEL = load_model(MODEL_FILE)
    SCALER = joblib.load(SCALER_FILE)
    if not hasattr(SCALER, 'feature_names_in_'): print("⚠️ Scaler does not contain feature list.")
    print("✅ Bybit CNN+Transformer model and scaler loaded successfully.")
except Exception as e: print(f"❌ Loading error: {e}")

# --- PREDICTION FUNCTION ---
def get_prediction_with_indicators(klines: list) -> tuple:
    if MODEL is None or SCALER is None: print("❌ Model/scaler not loaded."); return ([0.5, 0.5], {})
    required_length = 200 + SEQUENCE_LENGTH # With reserve
    if len(klines) < required_length: print(f"⚠️ Not enough data ({len(klines)} < {required_length})"); return ([0.5, 0.5], {})

    # 2. DataFrame (adapted for Bybit klines/stubs)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'turnover', # Use 'turnover' as 'quote_asset_volume'
        'n_trades_stub', 'taker_base_stub', 'taker_quote_stub', 'ignore_stub'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms'); df.set_index('open_time', inplace=True)
    
    # --- Important: Create columns that feature_engineering expects ---
    df['quote_asset_volume'] = df['turnover']
    # df['number_of_trades'] = 0 # These features are truncated
    # df['taker_buy_base_asset_volume'] = 0 # These features are truncated

    numeric_cols = ['open','high','low','close','volume','quote_asset_volume']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if len(df) < 200 + SEQUENCE_LENGTH: print(f"⚠️ Not enough data ({len(df)}) after cleaning OHLCV."); return ([0.5, 0.5], {})
    
    df = df.copy()

    # 3. Feature generation (Using training.feature_engineering)
    print("Generating features (prediction)...")
    try:
        from training.feature_engineering import generate_features
        df_features = generate_features(df)
        print("✅ Features generated.")
    except Exception as feature_err:
        print(f"❌ FEATURE GENERATION ERROR: {feature_err}"); traceback.print_exc(limit=3)
        return ([0.5, 0.5], {})

    # 4. Processing NaN with ffill and bfill
    print(f"Rows BEFORE ffill/bfill: {len(df_features)}")
    df_processed = df_features.copy()
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.ffill(inplace=True); df_processed.bfill(inplace=True)
    print(f"Rows AFTER ffill + bfill: {len(df_processed)}")

    # 5. Check for NaN and data sufficiency
    try: feature_list_from_scaler = list(SCALER.feature_names_in_)
    except AttributeError: print("❌ No feature list in scaler."); return ([0.5, 0.5], {})
    cols_to_check_nan = list(set(feature_list_from_scaler + ['EMA_9', 'EMA_26', 'RSI_14']))
    cols_to_check_nan = [col for col in cols_to_check_nan if col in df_processed.columns]
    
    if not cols_to_check_nan: print("❌ No columns to check for NaN."); return ([0.5, 0.5], {})

    if df_processed[cols_to_check_nan].isnull().values.any():
        nan_cols = df_processed[cols_to_check_nan].isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"❌ NaN AFTER ffill+bfill in required columns: {nan_cols.index.tolist()}")
        return ([0.5, 0.5], {})

    if len(df_processed) < SEQUENCE_LENGTH:
         print(f"⚠️ Not enough rows ({len(df_processed)}) for window {SEQUENCE_LENGTH} after NaN.")
         return ([0.5, 0.5], {})

    # 6. Preparing data for model
    last_seq = df_processed.tail(SEQUENCE_LENGTH)
    missing = [f for f in feature_list_from_scaler if f not in last_seq.columns]
    if missing: print(f"❌ Missing features for scaler: {missing}"); return ([0.5, 0.5], {})

    try:
        data_to_scale = last_seq[feature_list_from_scaler]
        if data_to_scale.isnull().values.any(): print("❌ NaN before scaling!"); return ([0.5, 0.5], {})
        scaled = SCALER.transform(data_to_scale)
    except Exception as scale_err: print(f"❌ Scaling error: {scale_err}"); return ([0.5, 0.5], {})

    X_input = np.expand_dims(scaled, axis=0)

    # 7. Model prediction
    try:
        p_long = MODEL.predict(X_input, verbose=0)[0][0]
        p_short = 1.0 - p_long; prediction = [p_short, p_long]
    except Exception as predict_err: print(f"❌ Prediction error: {predict_err}"); return ([0.5, 0.5], {})

    # 8. Collecting indicators for filter
    last_row_for_indicators = df_processed.iloc[-1]
    indicators = {
        "ema_fast": last_row_for_indicators.get('EMA_9'),
        "ema_slow": last_row_for_indicators.get('EMA_55'), # Use 55 for filter
        "rsi": last_row_for_indicators.get('RSI_14')
    }

    if any(v is None or pd.isna(v) for v in indicators.values()):
        print("⚠️ NaN in filter indicators (AFTER get).")
        return ([0.5, 0.5], {})

    # 9. Return result
    print("✅ Prediction and indicators obtained successfully.")
    return prediction, indicators