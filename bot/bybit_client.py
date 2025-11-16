# bot/bybit_client.py

import pandas as pd
import pandas_ta as ta
import time
from pybit.unified_trading import HTTP # Import HTTP client
from . import config # Import config for accessing settings
import traceback # For error output
import math # For quantity rounding

class BybitClient:
    def __init__(self, activation_manager=None):
        
        self._activation_manager = activation_manager
        
        self.client = HTTP(
            testnet=config.USE_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
        )
        print(f"✅ Bybit Client initialized (Testnet: {config.USE_TESTNET}).")
        self._set_initial_settings()
    
    def set_activation_manager(self, activation_manager):
        
        self._activation_manager = activation_manager
    
    def _check_activation(self):
        
        if self._activation_manager and not self._activation_manager.is_activated():
            raise RuntimeError(
                "❌ CRITICAL ERROR: Activation required to perform operation. "
                "Use /start to activate."
            )

    def _set_initial_settings(self):
        """Sets initial settings (ISOLATED margin mode)."""
        try:
            print(f"Attempting to set ISOLATED margin mode (tradeMode=1) for {config.PAIR}...")
            
            # --- FIX: Commented out due to UTA error (100028) ---
            # This command (switch_margin_mode) does not work for Unified Trading Accounts (UTA).
            # We will set the mode and leverage in the open_order function.
            
            # self.client.switch_margin_mode(
            #      category="linear", # For USDT Perpetual
            #      symbol=config.PAIR,
            #      tradeMode=1, # 1 = Isolated
            #      buyLeverage="1", 
            #      sellLeverage="1" 
            # )
            # print("✅ ISOLATED margin mode set (or was already set).")
            print("ℹ️ Skipping _set_initial_settings: (for Unified Account UTA). Mode will be set during trading.")
            # --- END OF FIX ---

        except Exception as e_margin:
            if "Same leverage" in str(e_margin) or "the same margin mode" in str(e_margin):
                 print("Margin mode is already ISOLATED.")
            else:
                 print(f"⚠️ Margin mode setup error: {e_margin}")

    def get_klines(self, limit=300):
        """Gets historical klines (candles) from Bybit."""
        self._check_activation()
        try:
            response = self.client.get_kline(
                category="linear", # USDT Perpetual
                symbol=config.PAIR,
                interval=config.INTERVAL, # Interval from config ('60')
                limit=limit
            )
            if response and response.get('retCode') == 0 and response.get('result', {}).get('list'):
                 klines_list = response['result']['list']
                 klines_list.reverse()
                 formatted_klines = []
                 for k in klines_list:
                      ts, o, h, l, c, v, turnover = k
                      formatted_klines.append([
                           int(ts), str(o), str(h), str(l), str(c), str(v),
                           int(ts) + (int(config.INTERVAL)*60*1000) -1, # Approximate close_time
                           str(turnover), 0, '0', '0', '0' # Placeholders
                      ])
                 return formatted_klines
            else:
                print(f"⚠️ Error getting klines from Bybit: {response.get('retMsg', 'No data')}")
                return []
        except Exception as e_klines:
            print(f"❌ Unknown error when getting klines: {e_klines}"); traceback.print_exc(limit=2)
            return []

    def calculate_leverage(self, df_processed: pd.DataFrame) -> int:
        """Calculates leverage (20x-50x) based on ATR."""
        ATR_COLUMN_NAME = f'ATR_{config.ATR_PERIOD}' # 'ATR_14'
        if df_processed.empty or ATR_COLUMN_NAME not in df_processed.columns or 'close' not in df_processed.columns: return 20
        atr_series = df_processed[ATR_COLUMN_NAME].dropna(); close_series = df_processed['close'].dropna()
        if atr_series.empty or close_series.empty: return 20
        atr_val = atr_series.iloc[-1]; close_price = close_series.iloc[-1]
        if pd.isna(atr_val) or pd.isna(close_price) or close_price <= 0 or atr_val <= 0: return 20
        volatility_percent = (atr_val / close_price) * 100
        print(f"Leverage calculation: Volatility={volatility_percent:.2f}%")
        if volatility_percent > 1.0: leverage = 20
        elif volatility_percent > 0.5: leverage = 35
        else: leverage = 50
        print(f"Selected leverage: {leverage}x")
        return leverage

    def get_balance(self, asset='USDT') -> float:
        """Gets the total balance of the derivatives wallet."""
        self._check_activation()
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED") # Or "CONTRACT"
            if response and response.get('retCode') == 0 and response.get('result', {}).get('list'):
                 acc_info = response['result']['list'][0]
                 for coin_balance in acc_info.get('coin', []):
                      if coin_balance.get('coin') == asset:
                           bal = float(coin_balance.get('walletBalance', '0'))
                           print(f"{asset} Balance (Unified): {bal:.2f}")
                           return bal
                 print(f"⚠️ Asset {asset} not found in Unified balance.")
                 return 0.0
            else:
                 print(f"⚠️ Error getting Bybit balance: {response.get('retMsg', 'No data')}")
                 return 0.0
        except Exception as e_bal:
             print(f"❌ Unknown error getting Bybit balance: {e_bal}"); traceback.print_exc(limit=2)
             return 0.0

    def get_quantity(self, balance: float, leverage: int, price: float) -> float:
        """Calculates quantity considering Bybit lot step."""
        if price <= 0 or balance <= 0 or leverage <= 0: return 0.0
        notional = balance * leverage * config.POSITION_SIZE_PERCENT / 100
        quantity = notional / price
        # --- ROUNDING FOR BYBIT ---
        step = config.QTY_STEP
        rounded_quantity = math.floor(quantity / step) * step
        decimals = 0
        if isinstance(step, float) and '.' in str(step):
             decimals = len(str(step).split('.')[-1])
        rounded_quantity = round(rounded_quantity, decimals)

        print(f"Quantity Calculation: Notional={notional:.2f}, Price={price:.3f}, Qty={quantity:.8f}, Step={step}, Rounded={rounded_quantity}")
        return rounded_quantity

    def open_order(self, side: str, leverage: int, tp_price: float = None, sl_price: float = None):
        """Opens a market order and sets TP/SL for Bybit."""
        self._check_activation()
        current_step = "Start of open_order"
        order_info = None
        actual_entry_price = None
        quantity_to_open = 0.0

        try:
            # --- FIX FOR UTA: Set margin and leverage BEFORE trading ---
            current_step = "Setting Margin/Leverage"
            print(f"⚙️ Setting ISOLATED margin mode and {leverage}x leverage for {config.PAIR}...")
            try:
                self.client.switch_margin_mode(
                     category="linear", symbol=config.PAIR, tradeMode=1, # 1 = Isolated
                     buyLeverage=str(leverage), sellLeverage=str(leverage)
                )
                print("✅ Margin mode and leverage set (Method 1).")
            except Exception as e_switch:
                print(f"⚠️ switch_margin_mode error (expected for UTA): {e_switch}")
                # Try second method (leverage only)
                try:
                    self.client.set_leverage(
                        category="linear", symbol=config.PAIR, 
                        buyLeverage=str(leverage), sellLeverage=str(leverage)
                    )
                    print(f"✅ Leverage {leverage}x set (Method 2).")
                except Exception as e_lev:
                     print(f"❌ Failed to set leverage: {e_lev}")
                     raise e_lev # Throw error if leverage is not set
            # --- END OF FIX ---

            # 2. Get price and calculate quantity
            current_step = "Quantity Calculation"
            ticker_info = self.client.get_tickers(category="linear", symbol=config.PAIR)
            if not (ticker_info and ticker_info.get('retCode') == 0 and ticker_info.get('result', {}).get('list')):
                 raise ValueError("Failed to get ticker for Mark Price.")
            entry_price_approx = float(ticker_info['result']['list'][0].get('markPrice', '0'))
            if entry_price_approx <= 0: raise ValueError("Invalid Mark Price.")

            balance = self.get_balance()
            if balance <= 0: raise ValueError(f"USDT Balance ({balance}) <= 0.")
            quantity_to_open = self.get_quantity(balance, leverage, entry_price_approx)
            if quantity_to_open <= 0: raise ValueError(f"Calculated quantity <= 0 ({quantity_to_open}).")

            # 3. Open market order
            current_step = "Creating Market Order"
            order_side = "Buy" if side == "BUY" else "Sell"
            print(f"📊 Opening {order_side} | Price ~{entry_price_approx:.3f} | Qty: {quantity_to_open} | Leverage: {leverage}x")
            response = self.client.place_order(
                category="linear",
                symbol=config.PAIR,
                side=order_side,
                orderType="Market",
                qty=str(quantity_to_open),
            )
            print(f"place_order response: {response}") 

            if response and response.get('retCode') == 0 and response.get('result', {}).get('orderId'):
                 order_info = response['result']
                 order_id = order_info['orderId']
                 print(f"✅ Order sent (ID: {order_id}). Waiting...")
                 time.sleep(3) 

                 # 4. Get actual entry price and set TP/SL
                 current_step = "Getting position and setting TP/SL"
                 position_info = self.get_open_positions()
                 if position_info:
                      actual_entry_price = float(position_info.get('avgPrice', entry_price_approx))
                      print(f"Position opened. Actual Entry Price: {actual_entry_price:.3f}")
                      if tp_price is not None and sl_price is not None:
                           self.set_trading_stop(tp_price, sl_price)
                      return order_info, actual_entry_price, quantity_to_open
                 else:
                      print("⚠️ Failed to get position information after opening.")
                      if tp_price is not None and sl_price is not None:
                           self.set_trading_stop(tp_price, sl_price)
                      return order_info, entry_price_approx, quantity_to_open
            else:
                 raise ValueError(f"Error creating Bybit order: {response.get('retMsg', 'No details')}")

        except ValueError as ve: print(f"❌ Validation error at step '{current_step}': {ve}"); return None, None, None
        except Exception as e: print(f"❌ Error at step '{current_step}': {e}"); traceback.print_exc(limit=2); return None, None, None

    def set_trading_stop(self, tp_price: float, sl_price: float):
        """Sets Take Profit and Stop Loss for an OPEN position."""
        try:
            print(f"Setting TP/SL for {config.PAIR}: TP={tp_price:.3f}, SL={sl_price:.3f}")
            response = self.client.set_trading_stop(
                category="linear",
                symbol=config.PAIR,
                takeProfit=str(round(tp_price, config.PRICE_PRECISION)),
                stopLoss=str(round(sl_price, config.PRICE_PRECISION)),
                tpslMode="Full", 
                tpTriggerBy="MarkPrice", 
                slTriggerBy="MarkPrice"  
            )
            print(f"set_trading_stop response: {response}") 
            if response and response.get('retCode') == 0:
                 print("✅ TP/SL successfully set.")
            else:
                 print(f"⚠️ Error setting Bybit TP/SL: {response.get('retMsg', 'No details')}")

        except Exception as e_ts:
            print(f"❌ Unknown error when setting TP/SL: {e_ts}")
            traceback.print_exc(limit=2)

    def get_open_positions(self):
        """Returns information about an open position for a pair or None."""
        self._check_activation()
        try:
            response = self.client.get_positions(category="linear", symbol=config.PAIR)
            if response and response.get('retCode') == 0 and response.get('result', {}).get('list'):
                 positions = response['result']['list']
                 if positions:
                      for pos in positions:
                           if float(pos.get('size', '0')) != 0:
                                return pos 
                 return None 
            else:
                 print(f"⚠️ Error getting Bybit position: {response.get('retMsg', 'No data')}")
                 return None
        except Exception as e_pos:
            print(f"❌ Unknown error when getting position: {e_pos}")
            return None

    def get_tp_sl_orders(self):
        """Gets information about set TP/SL from POSITION DATA."""
        position_info = self.get_open_positions()
        if position_info:
             tp_price = position_info.get('takeProfit')
             sl_price = position_info.get('stopLoss')
             tp_dict = {'triggerPrice': float(tp_price)} if tp_price and float(tp_price) > 0 else None
             sl_dict = {'triggerPrice': float(sl_price)} if sl_price and float(sl_price) > 0 else None
             if tp_dict: tp_dict['stopPrice'] = tp_dict['triggerPrice']
             if sl_dict: sl_dict['stopPrice'] = sl_dict['triggerPrice']
             return tp_dict, sl_dict
        return None, None

    def close_open_position(self):
        """Closes the current open position at market price."""
        self._check_activation()
        position = self.get_open_positions()
        if not position: return False, "No open positions."
        try:
            pos_size = float(position.get('size','0'))
            if pos_size == 0: return False, "Position already closed (Size=0)."
            
            side = "Sell" if pos_size > 0 else "Buy"
            quantity_to_close = abs(pos_size) 
            
            print(f"Closing position {side} | Qty: {quantity_to_close}")
            print("Canceling TP/SL before closing...")
            try:
                 self.client.set_trading_stop(
                      category="linear", symbol=config.PAIR,
                      takeProfit="0", stopLoss="0"
                 )
                 time.sleep(0.5)
            except Exception as cancel_err:
                 print(f"⚠️ Failed to cancel TP/SL: {cancel_err}")

            response = self.client.place_order(
                category="linear",
                symbol=config.PAIR,
                side=side,
                orderType="Market",
                qty=str(quantity_to_close),
                reduceOnly=True 
            )
            print(f"place_order response (close): {response}")

            if response and response.get('retCode') == 0:
                 order_id = response.get('result', {}).get('orderId', 'N/A')
                 print(f"Closing order sent (ID: {order_id}). Waiting...")
                 time.sleep(3)
                 final_pos = self.get_open_positions()
                 if final_pos is None or float(final_pos.get('size', '0')) == 0:
                      return True, "✅ Position successfully closed."
                 else:
                      return False, f"⚠️ Failed to confirm closing. Current Size: {final_pos.get('size')}"
            else:
                 return False, f"⚠️ Bybit error when sending closing order: {response.get('retMsg', 'No details')}"

        except Exception as e_close:
            print(f"❌ Unknown error when closing: {e_close}"); traceback.print_exc(limit=2)
            return False, f"Unknown error: {e_close}"

    def get_account_stats(self):
        """Collects statistics on balance and trades (PnL)."""
        self._check_activation()
        stats = {"balance": 0.0, "total": 0, "successful": 0, "unsuccessful": 0}
        try: # Get balance
            stats["balance"] = self.get_balance()
        except Exception as e_bal: print(f"⚠️ Error getting balance for statistics: {e_bal}")

        try: # Get PnL history
            response = self.client.get_transaction_log(
                 accountType="UNIFIED", # Or "CONTRACT"
                 category="linear",
                 type="RealisedPNL",
                 limit=500 
            )
            if response and response.get('retCode') == 0 and response.get('result', {}).get('list'):
                 trades = response['result']['list']
                 stats['total'] = len(trades)
                 stats['successful'] = sum(1 for t in trades if float(t.get('change', '0')) > 0)
                 stats['unsuccessful'] = stats['total'] - stats['successful']
            else:
                 print(f"⚠️ Error getting Bybit PnL history: {response.get('retMsg', 'No data')}")
            return stats
        except Exception as e_hist:
             print(f"⚠️ Error getting PnL history: {e_hist}")
             return stats