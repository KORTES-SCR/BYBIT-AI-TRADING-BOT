# bot/config.py

import os
import json
from dotenv import load_dotenv
# --- Removing Client import from binance ---
# from binance.client import Client

# Load environment variables from .env in the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- File paths (FIXED NAMES) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_bybit_1h.keras') # <-- FIXED
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_bybit_1h.pkl')         # <-- FIXED
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_bybit_1h.parquet') # <-- FIXED
SETTINGS_FILE = os.path.join(ROOT_DIR, 'settings.json')

# --- Bybit API ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
# --- Determine if Testnet is used ---
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
BYBIT_MAINNET_URL = "https://api.bybit.com"

# --- Telegram Bot (no changes) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Trading Settings ---
PAIR = os.getenv("PAIR", "SOLUSDT") # Bybit USDT Perpetual
# --- Bybit uses '60' for 1H ---
INTERVAL = '60' 
# --- PRECISION FOR BYBIT SOLUSDT ---
QTY_STEP = float(os.getenv("QTY_STEP", "0.01")) # Quantity change step
PRICE_PRECISION = int(os.getenv("PRICE_PRECISION", "3")) # Number of decimal places for price

# --- Position Size Settings ---
POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "5"))

# --- Model and Filter Settings (no changes) ---
SEQUENCE_LENGTH = 48
LONG_THRESHOLD = 0.7; SHORT_THRESHOLD = 0.3
RSI_MAX_LONG = 70; RSI_MIN_SHORT = 30
ATR_TP_MULTIPLIER = 2.5; ATR_SL_MULTIPLIER = 1.5; ATR_PERIOD = 14

# --- Local LLM Settings (no changes) ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"; LLM_MODEL_NAME = "mistral"; LLM_CONFIDENCE_THRESHOLD = 0.7

# --- Dynamic Settings (no changes) ---
def load_settings():
    defaults = {"TRADE_AMOUNT": 25.0, "TAKE_PROFIT": 2.5, "STOP_LOSS": 1.5}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: settings = json.load(f)
            for key, default_val in defaults.items():
                 if key not in settings: settings[key] = default_val
            return settings
        except Exception as e:
            print(f"⚠️ settings.json error: {e}. Using defaults.")
            return defaults
    else:
        print("settings.json not found. Using defaults."); save_settings(defaults); return defaults

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f, indent=2)
    except Exception as e: print(f"⚠️ Error saving settings.json: {e}")

settings = load_settings(); TRADE_AMOUNT = settings.get("TRADE_AMOUNT")
TAKE_PROFIT = settings.get("TAKE_PROFIT"); STOP_LOSS = settings.get("STOP_LOSS")

# --- ❗️ TRUNCATED FEATURE LIST (Bybit doesn't provide Taker Volume) ---
FEATURE_LIST = [
    'open', 'high', 'low', 'close', 'volume', # 'close' is not a feature
    'return_1h', 'return_3h', 'return_5h', 'return_15h', 'return_60h',
    'EMA_9', 'EMA_21', 'EMA_55', 'EMA_100', 'EMA_200', 'MA_50', 'MA_100',
    'RSI_14', 'RSI_28',
    'MACD', 'MACD_h', 'MACD_s', 'ATR_14',
    'STOCHRSIk', 'STOCHRSId',
    'BBU', 'BBL', 'BBB', 'STD_20',
    'volume_change', # 'taker_buy_ratio', # Cannot be calculated
    'OBV', 'MFI_14',
    'hour', 'weekday',
    'rolling_mean_6h', 'rolling_std_6h', 'rolling_std_48h',
    'rolling_skew_24h', 'rolling_skew_48h',
    'rolling_kurt_24h', 'rolling_kurt_48h'
]