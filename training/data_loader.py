# training/data_loader.py
import pandas as pd
from pybit.unified_trading import HTTP # Import HTTP client
import os
from dotenv import load_dotenv
import time
import traceback

def download_data(pair, interval, data_points, api_key, api_secret, use_testnet, cache_file=None):
    """Downloads historical data from Bybit, uses cache."""
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            if len(df) >= data_points:
                 print("Data successfully loaded from cache.")
                 if not isinstance(df.index, pd.DatetimeIndex): # Check index
                      if 'open_time' in df.columns:
                           df['open_time'] = pd.to_datetime(df['open_time'])
                           df.set_index('open_time', inplace=True)
                      elif df.index.name == 'open_time':
                           df.index = pd.to_datetime(df.index)
                      else:
                           print("Failed to restore DatetimeIndex, reloading...");
                           return download_fresh_data(pair, interval, data_points, api_key, api_secret, use_testnet, cache_file)
                 df.sort_index(inplace=True)
                 return df.iloc[-data_points:] # Return last N
            else:
                 print(f"Not enough data in cache ({len(df)} < {data_points}), reloading...")
        except Exception as e:
            print(f"Error reading cache {cache_file}: {e}. Reloading...")
        try: os.remove(cache_file)
        except OSError: pass

    return download_fresh_data(pair, interval, data_points, api_key, api_secret, use_testnet, cache_file)

def download_fresh_data(pair, interval, data_points, api_key, api_secret, use_testnet, cache_file=None):
    """Downloads fresh data from Bybit."""
    print(f"Downloading {data_points} candles for {pair} {interval} from Bybit...")
    
    client = HTTP(
        testnet=use_testnet,
        api_key=api_key,
        api_secret=api_secret
    )

    klines_all = []
    end_ts = None # Bybit uses 'end' (older data)
    limit_per_req = 1000 # Bybit allows up to 1000

    while len(klines_all) < data_points:
        fetch_limit = min(limit_per_req, data_points - len(klines_all))
        print(f"Requesting {fetch_limit} candles...", end="")
        try:
            params = {'category': "linear", 'symbol': pair, 'interval': interval, 'limit': fetch_limit}
            if end_ts:
                 params['end'] = end_ts # Request candles BEFORE this timestamp (in ms)

            response = client.get_kline(**params)

            if response and response.get('retCode') == 0 and response.get('result', {}).get('list'):
                klines = response['result']['list']
                print(f" Received {len(klines)}.")
                if not klines:
                    print("No more data (history limit reached)."); break
                
                # Data comes [newest...oldest], we need [oldest...newest]
                klines.reverse()
                
                # --- IMPORTANT: Remove duplicate ---
                if klines_all and klines_all[0][0] == klines[-1][0]:
                     klines_all = klines_all[1:] # Remove the first (newest) candle from the old list

                klines_all = klines + klines_all # Add to the beginning
                
                # Set end_ts for the next request
                oldest_kline_ts = int(klines[0][0]) # ts of the oldest candle
                # Step back by 1 interval (in ms) to avoid getting the same candle
                interval_ms = int(interval) * 60 * 1000 
                end_ts = oldest_kline_ts - interval_ms 
                
                time.sleep(0.5) # Pause (Bybit API limit: 10 req/sec)
            else:
                 print(f"\n⚠️ Bybit API error: {response.get('retMsg', 'No data')}. Retry in 5 sec...")
                 time.sleep(5)
        
        except Exception as e:
            print(f"\n❌ Unknown error during download: {e}. Retry in 5 sec...")
            traceback.print_exc(limit=2)
            time.sleep(5)
            
    print(f"Total downloaded {len(klines_all)} candles.")
    if not klines_all: return pd.DataFrame()

    # --- BYBIT DATA ADAPTATION ---
    # Bybit columns: [ts, open, high, low, close, volume, turnover]
    df = pd.DataFrame(klines_all, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    
    # --- CREATE PLACEHOLDER COLUMNS (Because this data is not in Bybit klines) ---
    df['quote_asset_volume'] = df['turnover'] # Use turnover
    df['number_of_trades'] = 0 # Placeholder
    df['taker_buy_base_asset_volume'] = 0 # Placeholder
    df['taker_buy_quote_asset_volume'] = 0 # Placeholder
    df['close_time'] = pd.to_numeric(df['open_time']) + (int(interval)*60*1000) - 1 # Approx.
    df['ignore'] = 0 # Placeholder
    
    # --- Formatting ---
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='first')]; df.sort_index(inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.iloc[-data_points:] # Take last N

    if cache_file:
        try:
            print(f"Saving data to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_parquet(cache_file)
        except Exception as e: print(f"Error saving cache: {e}")

    return df

# --- Block for test run ---
if __name__ == '__main__':
    # Specify path to .env one level UP (in project root)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path) 
    
    API_KEY = os.getenv("BYBIT_API_KEY")
    API_SECRET = os.getenv("BYBIT_API_SECRET")
    USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')
    PAIR = "SOLUSDT"
    INTERVAL = '60' # 1 hour
    DATA_POINTS = 2000 # ~3 months
    CACHE_DIR = "../data" # Data folder one level up
    CACHE_FILE = os.path.join(CACHE_DIR, f"{PAIR}_{INTERVAL}_cache.parquet")
    
    df_data = download_data(PAIR, INTERVAL, DATA_POINTS, API_KEY, API_SECRET, USE_TESTNET, CACHE_FILE)
    
    if not df_data.empty:
        print("\nExample of loaded data (beginning):")
        print(df_data.head())
        print("\nExample of loaded data (end):")
        print(df_data.tail())
        print(f"\nDataFrame size: {df_data.shape}")
        # Check for gaps in index
        time_diffs = df_data.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=int(INTERVAL))
        if time_diffs.max() > expected_interval * 1.1: # Allow small slack
             print(f"\n⚠️ WARNING: Data gaps detected! Max interval: {time_diffs.max()}")
    else:
        print("Failed to load data.")