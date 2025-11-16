# bot/main.py

import asyncio
import requests
import pandas as pd
import numpy as np # <-- Added import
import traceback
from datetime import datetime, timezone, timedelta # <-- Added import
from . import config # Common settings
from .bybit_client import BybitClient # <-- CHANGE
from .telegram_handler import TelegramHandler
from .activation import ActivationManager
from . import model_predictor # Module for predictions
# Using feature_engineering module from training folder
try:
    from training import feature_engineering
except ImportError:
    # If running via `python bot/main.py`, not `python -m bot.main`
    # May need to add project root to sys.path
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from training import feature_engineering
    print("⚠️ Failed to import training.feature_engineering by standard path, used workaround.")
except Exception as e_imp:
     print(f"❌ Critical error: Failed to import training.feature_engineering: {e_imp}")
     exit()


from . import filter_logic # Module for filter
from . import strategy_logic # Module for TP/SL
from . import local_llm # Module for LLM

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.in_position = False
        self.main_loop = asyncio.get_event_loop()
        self.current_trade_open_time = None # Timestamp in ms
        self.activation_manager = None  # Will be set on first activation
        self.bybit = BybitClient() # <-- CHANGE
        self.telegram = TelegramHandler(
            main_logic_callback=self.handle_command,
            bybit_client=self.bybit, # <-- CHANGE (passing bybit_client)
            get_bot_state_callback=lambda: self.is_running,
            trading_bot=self  # Pass reference to self for setting activation_manager
        )
        if not model_predictor.load_model_and_scaler():
             print("❌ Critical error: Failed to load model/scaler.")
             # Can add log sending to Telegram if critical
             # asyncio.run(self.telegram.send_log("❌ CRITICAL ERROR: Failed to load model/scaler."))
             # exit()
    
    def set_activation_manager(self, activation_manager):
        
        self.activation_manager = activation_manager
        self.bybit.set_activation_manager(activation_manager)

    def handle_command(self, command):
        if command == 'start_robot':
            if not self.is_running:
                self.is_running = True
                # --- FIXED INDENTATION ---
                asyncio.run_coroutine_threadsafe(self.telegram.send_log("🟢 Robot started! Starting analysis..."), self.main_loop)
        elif command == 'stop_robot':
            if self.is_running:
                self.is_running = False
                # --- FIXED INDENTATION ---
                asyncio.run_coroutine_threadsafe(self.telegram.send_log("🔴 Robot stopped by command."), self.main_loop)

    async def run_trading_cycle(self):
        if not self.is_running: return
        
        
        if not self.activation_manager or not self.activation_manager.is_activated():
            print("⚠️ Trading cycle skipped: activation required")
            await self.telegram.send_log("⚠️ Activation required for bot operation. Use /start")
            return

        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Entering run_trading_cycle...")
        current_step = "Start of cycle" # For debugging
        try:
            # 1. Checking current position
            current_step = "Position check"
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Calling get_open_positions...")
            open_position_info = self.bybit.get_open_positions() # <-- CHANGE
            currently_in_position = open_position_info is not None
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] get_open_positions completed. Result: {'Position exists' if currently_in_position else 'No position'}")

            # 2. Logging PnL on close
            current_step = "PnL Logging"
            log_pnl_now = False
            if self.in_position and not currently_in_position:
                print("[Close check] API reported no position. Checking PnL...")
                log_pnl_now = True
                self.in_position = False # Update INTERNAL status
            elif not self.in_position and currently_in_position:
                 print("Open position detected."); self.in_position = True
            
            if log_pnl_now:
                await self.telegram.send_log("🔄 Position was closed. Checking result...")
                try:
                    # --- CHANGE: Using get_account_stats for Bybit PnL ---
                    response = self.bybit.client.get_transaction_log(accountType="UNIFIED", category="linear", type="RealisedPNL", limit=1)
                    if response and response.get('retCode') == 0 and response.get('data', {}).get('list'):
                         pnl = float(response['data']['list'][0].get('change', '0'))
                         await self.telegram.send_log(f"✅ PROFIT: +{pnl:.2f} USDT" if pnl > 0 else f"❌ LOSS: {pnl:.2f} USDT")
                    else:
                        await self.telegram.send_log("Failed to get PnL from history.")
                except Exception as pnl_err:
                    await self.telegram.send_log(f"⚠️ Error getting PnL: {pnl_err}")

            if self.in_position:
                 print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] In position, skipping cycle."); return

            # --- Logic for 1H CNN + Transformer (Bybit) ---
            
            # 3. Get historical data
            current_step = "Loading Klines"
            required_candles = 300 # ~12 days of hourly candles
            await self.telegram.send_log(f"Getting quotes ({config.INTERVAL}, {required_candles} candles)...")
            klines = self.bybit.get_klines(limit=required_candles) # <-- CHANGE
            if len(klines) < config.SEQUENCE_LENGTH + 200: await self.telegram.send_log(f"⚠️ Not enough Klines ({len(klines)}) for analysis."); return

            # 4. Create DataFrame (adapted for Bybit klines/stubs)
            current_step = "Creating DataFrame"
            df_raw = pd.DataFrame(klines, columns=[ 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'turnover', 'n_trades_stub', 'taker_base_stub', 'taker_quote_stub', 'ignore_stub'])
            df_raw['open_time'] = pd.to_datetime(df_raw['open_time'], unit='ms'); df_raw.set_index('open_time', inplace=True)
            df_raw['quote_asset_volume'] = df_raw['turnover'] # Analog
            # Add stubs for features that don't exist but may be in feature_list
            df_raw['number_of_trades'] = 0
            df_raw['taker_buy_base_asset_volume'] = 0
            
            numeric_cols = ['open','high','low','close','volume','quote_asset_volume']
            for col in numeric_cols: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            df_raw.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            if len(df_raw) < config.SEQUENCE_LENGTH + 200: await self.telegram.send_log("⚠️ Not enough data after cleaning OHLCV."); return

            # 5. Generate features
            current_step = "Feature generation"
            df_features = feature_engineering.generate_features(df_raw) # Using truncated function

            # 6. Processing NaN
            current_step = "NaN Processing"
            df_processed = df_features.copy()
            df_processed.replace([np.inf, -np.inf], np.nan, inplace=True) # Using np
            df_processed.ffill(inplace=True); df_processed.bfill(inplace=True)
            try: feature_list_for_check = list(model_predictor.SCALER.feature_names_in_)
            except AttributeError: print("❌ No feature list in scaler."); await self.telegram.send_log("Critical error: no feature list."); return
            
            # Columns to check for NaN = scaler features + filter features
            cols_to_check_nan = list(set(feature_list_for_check + ['EMA_9', 'EMA_21', 'EMA_55', 'EMA_100', 'RSI_14', 'ATR_14', 'volume']))
            cols_to_check_nan = [col for col in cols_to_check_nan if col in df_processed.columns]
            if not cols_to_check_nan: await self.telegram.send_log("⚠️ Error: No columns to check for NaN."); return
            
            # Check NaN
            if df_processed[cols_to_check_nan].isnull().values.any():
                 nan_cols = df_processed[cols_to_check_nan].isnull().sum()
                 nan_cols = nan_cols[nan_cols > 0]
                 await self.telegram.send_log(f"⚠️ Error: NaN remained after processing in {nan_cols.index.tolist()}. Skipping.")
                 return
            if len(df_processed) < config.SEQUENCE_LENGTH: await self.telegram.send_log(f"⚠️ Not enough data ({len(df_processed)}) after NaN processing."); return

            # 7. Preparing data for MODEL
            current_step = "Preparing for model"
            last_seq_df = df_processed.tail(config.SEQUENCE_LENGTH)
            missing_scaler_features = [f for f in feature_list_for_check if f not in last_seq_df.columns]
            if missing_scaler_features: await self.telegram.send_log(f"❌ Error: Missing features for model: {missing_scaler_features}"); return
            if last_seq_df[feature_list_for_check].isnull().values.any(): await self.telegram.send_log("❌ Error: NaN in model data before scaling!"); return
            
            scaled_features_for_model = model_predictor.SCALER.transform(last_seq_df[feature_list_for_check])
            scaled_features_df = pd.DataFrame(scaled_features_for_model, index=last_seq_df.index, columns=feature_list_for_check)

            # 8. Get MODEL prediction
            current_step = "Model prediction"
            prediction = model_predictor.get_model_prediction(scaled_features_df)
            p_short, p_long = prediction[0], prediction[1]
            await self.telegram.send_log(f"🧠 CNN+Transformer Prediction: LONG {p_long:.2%}, SHORT {p_short:.2%}")

            # 9. Apply TECHNICAL FILTER
            current_step = "Technical filter"
            filter_result = filter_logic.apply_signal_filter(df_processed)
            if filter_result.get("error"): await self.telegram.send_log(f"⚠️ Filter error: {filter_result['error']}"); return
            long_confirmed = filter_result.get("long_confirmed", False)
            short_confirmed = filter_result.get("short_confirmed", False)
            
            # --- Send detailed filter log ---
            last_row = df_processed.iloc[-1]
            ema9=last_row.get('EMA_9',np.nan); ema21=last_row.get('EMA_21',np.nan); ema55=last_row.get('EMA_55',np.nan); ema100=last_row.get('EMA_100',np.nan)
            rsi14=last_row.get('RSI_14',np.nan); atr14=last_row.get('ATR_14',np.nan); volume=last_row.get('volume',np.nan)
            avg_atr = df_processed['ATR_14'].tail(20).mean() if len(df_processed) >= 20 else np.nan
            avg_volume = df_processed['volume'].tail(20).mean() if len(df_processed) >= 20 else np.nan
            
            # --- Using weakened filters 0.8/0.5 (as discussed) ---
            atr_threshold = avg_atr * 0.8 
            volume_threshold = avg_volume * 0.5
            
            # Check that thresholds are not NaN
            if pd.isna(atr_threshold) or pd.isna(volume_threshold):
                 await self.telegram.send_log("⚠️ Error: Failed to calculate avg_atr/avg_volume. Skipping.")
                 return

            detailed_log_msg = f"🔎 *Filter details:*\n"
            detailed_log_msg += f"  *Long:* {'✅' if long_confirmed else '❌'}\n"
            detailed_log_msg += f"    - EMA9 > EMA21: {'✅' if ema9 > ema21 else '❌'} ({ema9:.2f} > {ema21:.2f})\n"
            detailed_log_msg += f"    - EMA21 > EMA55: {'✅' if ema21 > ema55 else '❌'} ({ema21:.2f} > {ema55:.2f})\n"
            detailed_log_msg += f"    - RSI14 < {config.RSI_MAX_LONG}: {'✅' if rsi14 < config.RSI_MAX_LONG else '❌'} ({rsi14:.2f})\n"
            detailed_log_msg += f"    - ATR > {0.8:.1f}*Avg: {'✅' if atr14 > atr_threshold else '❌'} ({atr14:.3f} > {atr_threshold:.3f})\n"
            detailed_log_msg += f"    - Vol > {0.5:.1f}*Avg: {'✅' if volume > volume_threshold else '❌'} ({volume:.0f} > {volume_threshold:.0f})\n"
            detailed_log_msg += f"  *Short:* {'✅' if short_confirmed else '❌'}\n"
            detailed_log_msg += f"    - EMA9 < EMA21: {'✅' if ema9 < ema21 else '❌'} ({ema9:.2f} < {ema21:.2f})\n"
            detailed_log_msg += f"    - EMA21 < EMA55: {'✅' if ema21 < ema55 else '❌'} ({ema21:.2f} < {ema55:.2f})\n"
            detailed_log_msg += f"    - RSI14 > {config.RSI_MIN_SHORT}: {'✅' if rsi14 > config.RSI_MIN_SHORT else '❌'} ({rsi14:.2f})\n"
            detailed_log_msg += f"    - ATR > {0.8:.1f}*Avg: {'✅' if atr14 > atr_threshold else '❌'} ({atr14:.3f} > {atr_threshold:.3f})\n"
            detailed_log_msg += f"    - Vol > {0.5:.1f}*Avg: {'✅' if volume > volume_threshold else '❌'} ({volume:.0f} > {volume_threshold:.0f})"
            try:
                 await self.telegram.send_log(self.telegram.escape_markdown_v2(detailed_log_msg), parse_mode='MarkdownV2')

            except Exception as log_err: print(f"Failed to send detailed filter log: {log_err}")

            # 10. Decision mechanism
            current_step = "Decision mechanism"
            signal = "NONE"
            if p_long > config.LONG_THRESHOLD and long_confirmed: signal = "BUY"
            elif p_long < config.SHORT_THRESHOLD and short_confirmed: signal = "SELL"
            else: await self.telegram.send_log("...signal absent or not confirmed."); return

            # 11. (Optional) Request to LOCAL LLM
            current_step = "LLM Request"
            use_llm = False # Set to False to disable
            if use_llm and signal != "NONE":
                safe_filter_result = {"long_confirmed": bool(filter_result.get("long_confirmed", False)), "short_confirmed": bool(filter_result.get("short_confirmed", False))}
                indicator_keys = ['EMA_9','EMA_21','EMA_55','EMA_100','RSI_14','ATR_14','volume']
                last_indicators_series = df_processed.iloc[-1].get(indicator_keys)
                numeric_indicators = pd.to_numeric(last_indicators_series, errors='coerce'); rounded_indicators = numeric_indicators.round(3)
                indicators_dict = {k: (float(v) if pd.notna(v) else None) for k, v in rounded_indicators.to_dict().items()}
                model_pred_dict = {"LONG": float(p_long) if pd.notna(p_long) else None, "SHORT": float(p_short) if pd.notna(p_short) else None}

                llm_market_data = { "pair": config.PAIR, "timeframe": config.INTERVAL, "model_prediction": model_pred_dict, "filter_result": safe_filter_result, "indicators": indicators_dict, "current_signal": "LONG" if signal == "BUY" else "SHORT", "open_position": False }
                llm_response = local_llm.get_llm_decision(llm_market_data)
                if llm_response:
                     llm_action = llm_response.get("suggested_action", "HOLD").upper(); llm_confidence = llm_response.get("confidence", 0.0); llm_reason = llm_response.get("reason", "N/A")
                     await self.telegram.send_log(f"🤖 LLM ({config.LLM_MODEL_NAME}): {llm_action} (Conf: {llm_confidence:.0%}). Reason: {llm_reason}")
                     if (signal == "BUY" and llm_action == "LONG" and llm_confidence >= config.LLM_CONFIDENCE_THRESHOLD) or \
                        (signal == "SELL" and llm_action == "SHORT" and llm_confidence >= config.LLM_CONFIDENCE_THRESHOLD):
                         await self.telegram.send_log("👍 LLM confirms signal.")
                     else: await self.telegram.send_log("⚠️ LLM does NOT confirm signal. Trade canceled."); signal = "NONE"
                else: await self.telegram.send_log("⚠️ Failed to get response from LLM. Using Model+Filter signal.")

            # 12. Opening trade
            current_step = "Opening trade"
            if signal != "NONE":
                leverage = self.bybit.calculate_leverage(df_processed) # <-- CHANGE
                entry_price_approx = df_processed['close'].iloc[-1]
                tp_price, sl_price = strategy_logic.calculate_atr_stops(df_processed, entry_price_approx, signal == "BUY")
                if tp_price is None or sl_price is None: await self.telegram.send_log("❌ ERROR: Failed to calculate ATR TP/SL."); return

                # Log with % of balance, not old TRADE_AMOUNT
                await self.telegram.send_log(f"Opening {signal} trade on {config.POSITION_SIZE_PERCENT}% of balance with leverage {leverage}× | TP: ${tp_price:.3f}, SL: ${sl_price:.3f}")
                order, actual_entry_price, quantity = self.bybit.open_order(signal, leverage, tp_price, sl_price) # <-- CHANGE

                if order and actual_entry_price is not None:
                    asset_symbol = config.PAIR.replace("USDT", "")
                    await self.telegram.send_log(f"✅ Trade opened! {'LONG' if signal == 'BUY' else 'SHORT'} | {config.PAIR} | Entry: ${actual_entry_price:.3f} | Qty: {quantity} {asset_symbol}")
                    self.in_position = True
                else:
                    await self.telegram.send_log("❌ ERROR: Failed to open trade.")

        # --- RELIABLE ERROR HANDLING ---
        except requests.exceptions.ReadTimeout:
            error_message = "⏳ Network timeout (Binance/Bybit)."
            print(error_message)
            try:
                await self.telegram.send_log(error_message)
            except Exception as tg_e:
                print(f"TG ERR: {tg_e}")
        except requests.exceptions.ConnectionError as e:
            error_message = f"🔌 Network failure (Binance/Bybit): {e}."
            print(error_message)
            try:
                await self.telegram.send_log(error_message)
            except Exception as tg_e:
                print(f"TG ERR: {tg_e}")
        except Exception as e:
            error_message = f"🔥 Critical error at step '{current_step}': {e}"
            print(f"{error_message}\n--- TRACEBACK ---")
            traceback.print_exc(); print("--- END TRACEBACK ---")
            try:
                await self.telegram.send_log(f"🔥 Critical error (see console): {type(e).__name__} at '{current_step}'")
            except Exception as tg_err:
                print(f"TG ERR (critical): {tg_err}")
        # --- END OF ERROR HANDLING ---

        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] End of run_trading_cycle.")


    async def main(self):
        """Main bot cycle with waiting for next CANDLE (1H)
           and checking status EVERY MINUTE."""
        while True:
            try:
                if self.is_running:
                    now_utc = datetime.now(timezone.utc)

                    # --- Logic for waiting next HOURLY candle ---
                    next_hour = (now_utc + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    wait_seconds = (next_hour - now_utc).total_seconds()
                    wait_seconds += 5 # Small delay

                    print(f"Next 1H analysis at {next_hour.strftime('%Y-%m-%d %H:%M:%S UTC')}. Waiting {wait_seconds:.0f} sec (checking status every minute)...")

                    # Interruptible wait: check status every MINUTE
                    while wait_seconds > 0 and self.is_running:
                         sleep_interval = min(60, wait_seconds) # Wait 60 seconds or remainder
                         await asyncio.sleep(sleep_interval)
                         wait_seconds -= sleep_interval # Decrease by actual sleep time

                    if self.is_running:
                         print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}: Starting hourly candle analysis...")
                         await self.run_trading_cycle()
                    else:
                         print("Robot stopped during waiting.")
                         await asyncio.sleep(5)

                else: # If robot is stopped
                    await asyncio.sleep(5) # Check status every 5 seconds

            except Exception as loop_err:
                 print(f"Critical error in main asyncio loop: {loop_err}")
                 traceback.print_exc()
                 await asyncio.sleep(60) # Pause before restarting loop

async def start_bot():
    bot = TradingBot()
    await asyncio.gather(bot.telegram.start_bot(), bot.main())

if __name__ == "__main__":
    try: asyncio.run(start_bot())
    except KeyboardInterrupt: print("\nBot stopped manually.")
    except Exception as main_e: print(f"Critical startup error: {main_e}"); traceback.print_exc()
