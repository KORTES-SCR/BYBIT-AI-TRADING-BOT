# bot/filter_logic.py
import pandas as pd
import numpy as np
import traceback
from . import config  # Import config for RSI thresholds

def apply_signal_filter(df_processed: pd.DataFrame) -> dict:
    """
    Applies an advanced technical filter to the last row of processed data.
    Returns a dictionary with verification results for LONG and SHORT.
    """
    if df_processed.empty:
        return {"long_confirmed": False, "short_confirmed": False, "error": "No data for filter"}

    last_row = df_processed.iloc[-1]

    # --- Get indicator values ---
    ema9 = last_row.get('EMA_9', np.nan)
    ema21 = last_row.get('EMA_21', np.nan)
    ema55 = last_row.get('EMA_55', np.nan)
    ema100 = last_row.get('EMA_100', np.nan)
    rsi14 = last_row.get('RSI_14', np.nan)
    atr14 = last_row.get('ATR_14', np.nan)
    volume = last_row.get('volume', np.nan)

    # --- Average ATR and Volume ---
    avg_atr = np.nan
    avg_volume = np.nan
    if len(df_processed) >= 20:
        avg_atr = df_processed['ATR_14'].tail(20).mean()
        avg_volume = df_processed['volume'].tail(20).mean()

    # --- Check for NaN ---
    required_values = [ema9, ema21, ema55, ema100, rsi14, atr14, volume, avg_atr, avg_volume]
    if any(pd.isna(v) for v in required_values):
        print("⚠️ NaN in filter indicators:")
        print(f"  EMA9={ema9}, EMA21={ema21}, EMA55={ema55}, EMA100={ema100}")
        print(f"  RSI14={rsi14}, ATR14={atr14}, Volume={volume}")
        print(f"  AvgATR={avg_atr}, AvgVol={avg_volume}")
        return {"long_confirmed": False, "short_confirmed": False, "error": "NaN in filter data"}

    # --- "Aggressive" filter logic ---
    try:
        # --- CHANGE: Add multipliers ---
        atr_threshold = avg_atr * 0.8  # Require ATR > 80% of average
        volume_threshold = avg_volume * 0.5 # Require Volume > 50% of average

        long_filter_passed = (
            ema9 > ema21 and
            ema21 > ema55 and
            rsi14 < config.RSI_MAX_LONG and
            atr14 > atr_threshold and  # <-- Use atr_threshold threshold
            volume > volume_threshold # <-- Use volume_threshold threshold
        )


        short_filter_passed = (
            ema9 < ema21 and
            ema21 < ema55 and
            rsi14 > config.RSI_MIN_SHORT and
            atr14 > atr_threshold and # <-- Use atr_threshold threshold
            volume > volume_threshold # <-- Use volume_threshold threshold
        )

        # --- Log results ---
        print("--- Filter Results ---")
        # Calculate thresholds for log
        atr_threshold = avg_atr * 0.8
        volume_threshold = avg_volume * 0.5

        # Form log strings for Long
        log_long_ema = f"EMA9>21:{ema9>ema21}({ema9:.2f}>{ema21:.2f})"
        log_long_ema2 = f"EMA21>55:{ema21>ema55}({ema21:.2f}>{ema55:.2f})"
        log_long_rsi = f"RSI<{config.RSI_MAX_LONG}:{rsi14<config.RSI_MAX_LONG}({rsi14:.2f})"
        log_long_atr = f"ATR>0.8*avg:{atr14>atr_threshold}({atr14:.3f}>{atr_threshold:.3f})"
        log_long_vol = f"Vol>0.5*avg:{volume>volume_threshold}({volume:.0f}>{volume_threshold:.0f})"
        print(f"Long Filter -> {log_long_ema}, {log_long_ema2}, {log_long_rsi}, {log_long_atr}, {log_long_vol} => {long_filter_passed}")

        # Form log strings for Short
        log_short_ema = f"EMA9<21:{ema9<ema21}({ema9:.2f}<{ema21:.2f})"
        log_short_ema2 = f"EMA21<55:{ema21<ema55}({ema21:.2f}<{ema55:.2f})"
        log_short_rsi = f"RSI>{config.RSI_MIN_SHORT}:{rsi14>config.RSI_MIN_SHORT}({rsi14:.2f})"
        log_short_atr = f"ATR>0.8*avg:{atr14>atr_threshold}({atr14:.3f}>{atr_threshold:.3f})" # ATR condition is the same
        log_short_vol = f"Vol>0.5*avg:{volume>volume_threshold}({volume:.0f}>{volume_threshold:.0f})" # Volume condition is the same
        print(f"Short Filter -> {log_short_ema}, {log_short_ema2}, {log_short_rsi}, {log_short_atr}, {log_short_vol} => {short_filter_passed}")

        print("--------------------------")

        return {"long_confirmed": long_filter_passed, "short_confirmed": short_filter_passed}

    except Exception as e:
        print(f"❌ Error applying filter: {e}")
        traceback.print_exc(limit=2)
        return {"long_confirmed": False, "short_confirmed": False, "error": str(e)}
