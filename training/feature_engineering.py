# training/feature_engineering.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all NECESSARY features for the TRUNCATED Bybit/OKX model.
    (because taker_buy_volume and number_of_trades are missing)
    """
    print("Feature generation (Bybit/Truncated)...")
    df = df.copy() # Work with a copy to avoid ChainedAssignmentWarning

    # --- Returns ---
    try:
        print("- Calculating Returns...")
        for lag in [1, 3, 5, 15, 60]: # Use lags in interval units (1H)
             df[f'return_{lag}h'] = df['close'].pct_change(lag)
    except Exception as e:
        print(f"⚠️ Error calculating Returns: {e}. Columns will be NaN."); traceback.print_exc(limit=1)
        for lag in [1, 3, 5, 15, 60]: df[f'return_{lag}h'] = np.nan

    # --- EMA/MA ---
    try:
        print("- Calculating EMA/MA...")
        for length in [9, 21, 55, 100, 200]: df[f'EMA_{length}'] = ta.ema(df['close'], length=length)
        for length in [50, 100]: df[f'MA_{length}'] = ta.sma(df['close'], length=length)
    except Exception as e:
        print(f"⚠️ Error calculating EMA/MA: {e}. Columns will be NaN."); traceback.print_exc(limit=1)
        for length in [9, 21, 55, 100, 200]: df[f'EMA_{length}'] = np.nan
        for length in [50, 100]: df[f'MA_{length}'] = np.nan

    # --- Momentum ---
    try:
        print("- Calculating RSI, MACD, ATR, MFI, StochRSI...")
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        df.ta.rsi(length=28, append=True, col_names=('RSI_28',))
        df.ta.macd(append=True, col_names=('MACD', 'MACD_h', 'MACD_s'))
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))
        
        # MFI (requires high, low, close, volume)
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        if isinstance(mfi, pd.Series): df = df.join(mfi.rename('MFI_14'))
        else: df['MFI_14'] = np.nan
        
        # StochRSI (requires RSI_14)
        if 'RSI_14' in df.columns and df['RSI_14'].notna().sum() >= 20: # 14+3+3
             stoch_rsi = ta.stochrsi(df['RSI_14'], length=14, rsi_length=14, k=3, d=3)
             if isinstance(stoch_rsi, pd.DataFrame) and not stoch_rsi.empty:
                  # Search for columns by prefix
                  stoch_k_col = next((col for col in stoch_rsi.columns if col.startswith('STOCHRSIk')), None)
                  stoch_d_col = next((col for col in stoch_rsi.columns if col.startswith('STOCHRSId')), None)
                  # Use standard names to match config.FEATURE_LIST
                  if stoch_k_col: df['STOCHRSIk'] = stoch_rsi[stoch_k_col]
                  else: df['STOCHRSIk'] = np.nan
                  if stoch_d_col: df['STOCHRSId'] = stoch_rsi[stoch_d_col]
                  else: df['STOCHRSId'] = np.nan
             else:
                  print("⚠️ StochRSI returned incorrect result.")
                  df['STOCHRSIk'] = df['STOCHRSId'] = np.nan
        else: 
            print("⚠️ Skipping StochRSI: RSI_14 not calculated or insufficient data.")
            df['STOCHRSIk'] = df['STOCHRSId'] = np.nan
            
    except Exception as e:
        print(f"⚠️ Error calculating Momentum: {e}"); traceback.print_exc(limit=1)
        # Ensure columns exist if calculation failed
        for col in ['RSI_14', 'RSI_28', 'MACD', 'MACD_h', 'MACD_s', 'ATR_14', 'MFI_14', 'STOCHRSIk', 'STOCHRSId']:
             if col not in df: df[col] = np.nan

    # --- Volatility / Range ---
    try:
        print("- Calculating Bollinger Bands, STD...")
        # Calculation on cleaned data for reliability
        close_series = df['close'].dropna()
        if len(close_series) >= 20:
            bbands = ta.bbands(close_series, length=20, std=2)
            if bbands is not None and isinstance(bbands, pd.DataFrame) and not bbands.empty:
                # Dynamic name search
                bbu_col = next((col for col in bbands.columns if col.startswith('BBU_')), None)
                bbl_col = next((col for col in bbands.columns if col.startswith('BBL_')), None)
                bbb_col = next((col for col in bbands.columns if col.startswith('BBB_')), None)
                if bbu_col and bbl_col and bbb_col:
                    df = df.join(bbands[[bbu_col, bbl_col, bbb_col]]) # Join by index
                    df.rename(columns={bbu_col: 'BBU', bbl_col: 'BBL', bbb_col: 'BBB'}, inplace=True)
                else: raise ValueError("BBU/BBL/BBB columns not found")
            else: raise ValueError("ta.bbands() returned incorrect result")
        else: raise ValueError(f"Insufficient data ({len(close_series)}) for BBands")
        
        df['STD_20'] = df['close'].rolling(window=20).std()
    except Exception as bb_err:
        print(f"⚠️ Volatility error: {bb_err}"); traceback.print_exc(limit=1)
        df['BBU'] = df['BBL'] = df['BBB'] = df['STD_20'] = np.nan

    # --- Volume ---
    try:
        print("- Calculating Volume Features (OBV, Volume Change)...")
        # ❗️ Taker features REMOVED ❗️
        df['volume_change'] = df['volume'].pct_change()
        obv = ta.obv(df['close'], df['volume'])
        if isinstance(obv, pd.Series): df = df.join(obv) # Join OBV
        else: df['OBV'] = np.nan
    except Exception as vol_err:
        print(f"⚠️ Volume error: {vol_err}"); traceback.print_exc(limit=1)
        df['volume_change'] = df['OBV'] = np.nan

    # --- Time Features ---
    try:
        print("- Calculating Time Features...")
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
    except Exception as time_err:
        print(f"⚠️ Time error: {time_err}"); traceback.print_exc(limit=1)
        df['hour'] = df['weekday'] = np.nan

    # --- Rolling Stats (periods in hours for 1H timeframe) ---
    try:
        print("- Calculating Rolling Stats...")
        for window in [6, 48]: # 6 hours, 2 days
             df[f'rolling_std_{window}h'] = df['close'].rolling(window=window, min_periods=window//2).std()
        for window in [24, 48]: # 1 day, 2 days
             df[f'rolling_skew_{window}h'] = df['close'].rolling(window=window, min_periods=window//2).skew()
             df[f'rolling_kurt_{window}h'] = df['close'].rolling(window=window, min_periods=window//2).kurt()
        df['rolling_mean_6h'] = df['close'].rolling(window=6, min_periods=3).mean()
    except Exception as roll_err:
        print(f"⚠️ Rolling Stats error: {roll_err}"); traceback.print_exc(limit=1)
        # Ensure columns exist
        for window in [6, 48]: df[f'rolling_std_{window}h'] = np.nan
        for window in [24, 48]: df[f'rolling_skew_{window}h'] = df[f'rolling_kurt_{window}h'] = np.nan
        df['rolling_mean_6h'] = np.nan

    print("✅ Feature generation (Bybit/Truncated) completed.")
    return df

# --- Block for test run ---
if __name__ == '__main__':
    # Import data_loader for test
    from data_loader import download_data
    
    # Specify path to .env one level UP
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path) 
    
    API_KEY = os.getenv("BYBIT_API_KEY")
    API_SECRET = os.getenv("BYBIT_API_SECRET")
    USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')
    PAIR = "SOLUSDT"
    INTERVAL = '60' # 1 hour
    DATA_POINTS = 500 # Request 500 candles for test
    CACHE_DIR = "../data" # Data folder one level up
    CACHE_FILE = os.path.join(CACHE_DIR, f"{PAIR}_{INTERVAL}_cache.parquet")
    
    df_raw = download_data(PAIR, INTERVAL, DATA_POINTS, API_KEY, API_SECRET, USE_TESTNET, CACHE_FILE)
    
    if not df_raw.empty:
        print("\n--- Running generate_features for test ---")
        df_features = generate_features(df_raw)
        
        print("\nExample data with features (tail):")
        print(df_features.tail())
        print(f"\nDataFrame size with features: {df_features.shape}")
        
        print("\n--- NaN check after generation (BEFORE cleaning) ---")
        nan_counts = df_features.isnull().sum()
        print(nan_counts[nan_counts > 0].sort_values(ascending=False))
    else:
        print("Failed to load data for test.")