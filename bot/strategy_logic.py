# bot/strategy_logic.py
import pandas as pd
import numpy as np
from . import config # Import config for ATR multipliers

def calculate_atr_stops(df_processed: pd.DataFrame, entry_price: float, is_long: bool) -> tuple:
    """
    Calculates Take Profit and Stop Loss prices based on ATR.
    Returns (tp_price, sl_price) or (None, None) on error.
    """
    if df_processed.empty or 'ATR_14' not in df_processed.columns:
        print("❌ ATR Stops error: No data or ATR_14 column.")
        return None, None

    # Take the last NON-NaN ATR value
    atr_val = df_processed['ATR_14'].dropna().iloc[-1]

    if pd.isna(atr_val) or atr_val <= 0:
        print(f"❌ ATR Stops error: Invalid ATR value ({atr_val}).")
        return None, None

    print(f"TP/SL Calculation: Entry={entry_price:.3f}, ATR={atr_val:.3f}")

    if is_long:
        tp_price = entry_price + atr_val * config.ATR_TP_MULTIPLIER
        sl_price = entry_price - atr_val * config.ATR_SL_MULTIPLIER
    else: # Short
        tp_price = entry_price - atr_val * config.ATR_TP_MULTIPLIER
        sl_price = entry_price + atr_val * config.ATR_SL_MULTIPLIER

    # Round to the correct precision for the pair (3 digits for SOL)
    # TODO: Make precision dynamic based on config.PAIR
    price_precision = 3 if "SOL" in config.PAIR else 2 # Simplified definition

    tp_price = round(tp_price, price_precision)
    sl_price = round(sl_price, price_precision)

    print(f"Calculated stops: TP={tp_price}, SL={sl_price}")
    return tp_price, sl_price