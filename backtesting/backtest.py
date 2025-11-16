# backtesting/backtest.py

import backtrader as bt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv # For access to PAIR
import traceback # For debugging

# --- SETTINGS ---
# Look for .env one level up (if backtest.py is in backtesting/ folder)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

PAIR = os.getenv("PAIR", "SOLUSDT") # Get pair from .env
# Paths relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_bybit_1h.parquet')
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_bybit_1h.keras')
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_bybit_1h.pkl')

# SEQUENCE_LENGTH from train_model.py (need to know for strategy)
# Set manually, MAKE SURE IT MATCHES train_model.py!
SEQUENCE_LENGTH = 48
print(f"Using SEQUENCE_LENGTH={SEQUENCE_LENGTH}")


# --- Strategy and filter parameters ---
LONG_THRESHOLD = 0.7   # Threshold P(Long) for entering LONG
SHORT_THRESHOLD = 0.3  # Threshold P(Long) for entering SHORT (P(Short) > 0.7)
RSI_MAX_LONG = 70      # Max. RSI for entering LONG
RSI_MIN_SHORT = 30     # Min. RSI for entering SHORT
ATR_TP_MULTIPLIER = 2.5 # ATR multiplier for Take Profit
ATR_SL_MULTIPLIER = 1.5 # ATR multiplier for Stop Loss

# --- Backtest parameters ---
INITIAL_CASH = 10000.0
POSITION_SIZE_PERCENT = 90 # % of capital per trade
COMMISSION_RATE = 0.0006   # Bybit commission (Taker) ~0.06%
SLIPPAGE_PERCENT = 0.0002  # Slippage in % (0.02%)

# --- 1. Loading model and scaler ---
print("Loading model and scaler (Bybit)...")
model = None
scaler = None
feature_list = []
try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    if hasattr(scaler, 'feature_names_in_'):
        feature_list = list(scaler.feature_names_in_) # <-- Get TRUNCATED list
        print(f"Model expects {len(feature_list)} features.")
    else:
        print("❌ ERROR: Scaler does not contain feature list. Try retraining with newer sklearn version.")
        exit() # Exit if scaler is incorrect
except FileNotFoundError:
    print(f"❌ ERROR: Model file ('{MODEL_FILE}') or scaler file ('{SCALER_FILE}') not found.")
    exit() # Exit if files not found
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}"); traceback.print_exc(limit=2); exit() # Exit on other errors

# --- 2. Defining Backtrader strategy ---
class CnnTransformerStrategy(bt.Strategy):
    params = (
        ('seq_len', SEQUENCE_LENGTH),
        ('pred_threshold_long', LONG_THRESHOLD),
        ('pred_threshold_short', SHORT_THRESHOLD),
        ('feature_list', feature_list), # Pass truncated feature list
        ('atr_period', 14),
        ('atr_tp_mult', ATR_TP_MULTIPLIER),
        ('atr_sl_mult', ATR_SL_MULTIPLIER),
        ('rsi_max_long', RSI_MAX_LONG),
        ('rsi_min_short', RSI_MIN_SHORT),
    )

    def __init__(self):
        self.model = model
        self.scaler = scaler
        # --- Indicators ---
        # Make sure EMA200 is in the truncated feature list
        self.ema_fast = bt.indicators.EMA(self.data.close, period=9)
        self.ema_mid = bt.indicators.EMA(self.data.close, period=21)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=55)
        self.ema_trend1 = bt.indicators.EMA(self.data.close, period=100)
        # self.ema_trend2 = bt.indicators.EMA(self.data.close, period=200) # (If EMA200 is in the truncated list)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.volume_avg = bt.indicators.SimpleMovingAverage(self.data.volume, period=20)

        self.order = None # Active entry order
        self.buy_price = None
        self.buy_comm = None
        self.sl_order = None # Stop loss order reference
        self.tp_order = None # Take profit order reference

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0) # Use datetime for time
        print(f'{dt.strftime("%Y-%m-%d %H:%M:%S")}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return

        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            price = order.executed.price
            comm = order.executed.comm
            size = order.executed.size

            # Determine order type (entry, SL, TP)
            order_type = "ENTRY"
            if self.sl_order and order.ref == self.sl_order.ref:
                order_type = "STOP LOSS"
                self.sl_order = None
                if self.tp_order: self.cancel(self.tp_order); self.tp_order = None
            elif self.tp_order and order.ref == self.tp_order.ref:
                order_type = "TAKE PROFIT"
                self.tp_order = None
                if self.sl_order: self.cancel(self.sl_order); self.sl_order = None

            self.log(f'{action} EXECUTED ({order_type}), Price: {price:.3f}, Size: {size:.2f}, Comm: {comm:.3f}')

            if order_type == "ENTRY":
                 self.buy_price = price
                 self.buy_comm = comm
                 self.set_atr_stops(price, order.isbuy(), size)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: Status {order.Status[order.status]} Ref: {order.ref}')
            if self.sl_order and order.ref == self.sl_order.ref: self.sl_order = None
            if self.tp_order and order.ref == self.tp_order.ref: self.tp_order = None
            if self.order and order.ref == self.order.ref: self.order = None

        if self.order and order.ref == self.order.ref and order.status not in [order.Submitted, order.Accepted]:
             self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        self.log(f'TRADE CLOSED, PROFIT: NET {trade.pnlcomm:.2f}')

    def set_atr_stops(self, entry_price, is_long, size):
         """Sets Stop Loss and Take Profit orders based on ATR."""
         atr_val = self.atr[0]
         if np.isnan(atr_val) or atr_val <= 0:
              self.log("⚠️ Invalid ATR when setting stops!")
              return

         price_precision = config.PRICE_PRECISION # 3 for SOL
         
         if is_long:
              sl_price = round(entry_price - atr_val * self.p.atr_sl_mult, price_precision)
              tp_price = round(entry_price + atr_val * self.p.atr_tp_mult, price_precision)
              self.sl_order = self.sell(exectype=bt.Order.Stop, price=sl_price, size=size, oco=self.tp_order)
              self.tp_order = self.sell(exectype=bt.Order.Limit, price=tp_price, size=size, oco=self.sl_order)
              if self.sl_order: self.sl_order.addinfo(type="SL")
              if self.tp_order: self.tp_order.addinfo(type="TP")
              self.log(f'>>> LONG TP={tp_price:.3f}, SL={sl_price:.3f} set')
         else: # Short
              sl_price = round(entry_price + atr_val * self.p.atr_sl_mult, price_precision)
              tp_price = round(entry_price - atr_val * self.p.atr_tp_mult, price_precision)
              buy_size = abs(size)
              self.sl_order = self.buy(exectype=bt.Order.Stop, price=sl_price, size=buy_size, oco=self.tp_order)
              self.tp_order = self.buy(exectype=bt.Order.Limit, price=tp_price, size=buy_size, oco=self.sl_order)
              if self.sl_order: self.sl_order.addinfo(type="SL")
              if self.tp_order: self.tp_order.addinfo(type="TP")
              self.log(f'>>> SHORT TP={tp_price:.3f}, SL={sl_price:.3f} set')

    def next(self):
        # Check bars (EMA100 is longest in filter + window + 1)
        required_bars = max(100, self.p.seq_len) + 1
        if len(self.data) < required_bars: return
        if self.position or self.order: return

        # 1. Prepare data for model
        data_window_dict = {}; can_form_window = True
        for feature_name in self.p.feature_list:
             try:
                 line = self.data.lines.getlinealias(feature_name)
                 window_data = line.get(ago=-1, size=self.p.seq_len)
                 if len(window_data) != self.p.seq_len or np.isnan(window_data).any(): can_form_window = False; break
                 data_window_dict[feature_name] = window_data
             except KeyError: can_form_window = False; break
             except Exception: can_form_window = False; break
        if not can_form_window: return

        # Scale data
        try:
            X_input_df = pd.DataFrame(data_window_dict)[self.p.feature_list]
            X_input_scaled = self.scaler.transform(X_input_df)
            X_input_final = np.expand_dims(X_input_scaled, axis=0)
        except Exception as scale_err: self.log(f"Error preparing data: {scale_err}"); return

        # 2. Get model prediction
        try: p_long = self.model.predict(X_input_final, verbose=0)[0][0]
        except Exception as pred_err: self.log(f"Prediction error: {pred_err}"); return
        self.log(f'Forecast P(Long): {p_long:.3f}')

        # 3. Check filter (on current bar [0])
        ema9 = self.ema_fast[0]; ema21 = self.ema_mid[0]; ema55 = self.ema_slow[0]; ema100 = self.ema_trend1[0]
        rsi14 = self.rsi[0]; atr14 = self.atr[0]; vol = self.data.volume[0]; vol_avg = self.volume_avg[0]
        if any(np.isnan(v) for v in [ema9, ema21, ema55, ema100, rsi14, atr14, vol, vol_avg]): return

        atr_lookback_ready = len(self.atr) > 20
        # --- Use weakened filters (0.8 / 0.5) ---
        atr_threshold = self.atr[-20] * 0.8 if atr_lookback_ready else 0 
        volume_threshold = vol_avg * 0.5
        atr_condition = atr14 > atr_threshold if atr_lookback_ready else False
        volume_condition = vol > volume_threshold
        # --- END ---

        # (EMA 9>21 and 21>55) - to match recent logs
        long_filter = (ema9 > ema21 and ema21 > ema55 and rsi14 < self.p.rsi_max_long and atr_condition and volume_condition)
        short_filter = (ema9 < ema21 and ema21 < ema55 and rsi14 > self.p.rsi_min_short and atr_condition and volume_condition)

        # Log filters
        self.log(f'LONG Filter: {long_filter} (E9>21:{ema9>ema21}, E21>55:{ema21>ema55}, RSI<{self.p.rsi_max_long}:{rsi14<self.p.rsi_max_long}, ATR:{atr_condition}, Vol:{volume_condition})')
        self.log(f'SHORT Filter: {short_filter} (E9<21:{ema9<ema21}, E21<55:{ema21<ema55}, RSI>{self.p.rsi_min_short}:{rsi14>self.p.rsi_min_short}, ATR:{atr_condition}, Vol:{volume_condition})')

        # 4. Decision and entry
        signal = "NONE"
        if p_long > self.p.pred_threshold_long and long_filter: signal = "BUY"
        elif p_long < self.p.pred_threshold_short and short_filter: signal = "SELL"

        if signal != "NONE":
            self.log(f'>>> FINAL SIGNAL: {signal} <<<')
            if signal == "BUY": self.log(f'BUY CREATE, Price: {self.data.close[0]:.3f}'); self.order = self.buy()
            elif signal == "SELL": self.log(f'SELL CREATE, Price: {self.data.close[0]:.3f}'); self.order = self.sell()

# --- 3. Loading data for Backtrader ---
print(f"Loading data from {PROCESSED_DATA_FILE}...")
data = None # Initialize
try:
    df_bt = pd.read_parquet(PROCESSED_DATA_FILE)
    if not isinstance(df_bt.index, pd.DatetimeIndex):
         df_bt.index = pd.to_datetime(df_bt.index)
         if not isinstance(df_bt.index, pd.DatetimeIndex): raise ValueError("Not DatetimeIndex")
    df_bt.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}, inplace=True, errors='ignore')
    df_bt['openinterest'] = 0
    print(f"Loaded {len(df_bt)} rows of data.")

    # --- CREATING DATA CLASS WITH ADDITIONAL LINES ---
    if not feature_list: raise ValueError("Feature list (feature_list) is empty!")
    missing_data_cols = [f for f in feature_list if f not in df_bt.columns]
    if missing_data_cols: print(f"❌ ERROR: Columns missing in data: {missing_data_cols}"); exit()

    class PandasDataWithFeatures(bt.feeds.PandasData):
         lines = tuple(feature_list)
         params = tuple((f, -1) for f in feature_list) + \
                  (('open', -1), ('high', -1), ('low', -1), ('close', -1), ('volume', -1), ('openinterest', -1),)

    data = PandasDataWithFeatures(dataname=df_bt)
    print("✅ Data feed for Backtrader created.")

except FileNotFoundError: print(f"❌ File {PROCESSED_DATA_FILE} not found."); exit()
except Exception as e_data: print(f"❌ Data error for Backtrader: {e_data}"); traceback.print_exc(limit=2); exit()


# --- 4. Setting up and running Cerebro ---
print("Setting up Backtrader Cerebro...")
if data is None: print("❌ CRITICAL ERROR: 'data' was not created."); exit()

cerebro = bt.Cerebro(stdstats=False)
cerebro.adddata(data)
cerebro.addstrategy(CnnTransformerStrategy)
cerebro.broker.set_cash(INITIAL_CASH)
cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_SIZE_PERCENT)
cerebro.broker.setcommission(commission=COMMISSION_RATE)
cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PERCENT / 100.0) # Fixed

# Analyzers
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, factor=365, annualize=True, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

# --- 5. Running backtest ---
print("Running backtesting...")
try:
    results = cerebro.run()
    strat = results[0]
except Exception as run_err: print(f"❌ Backtest error: {run_err}"); traceback.print_exc(); exit()

# --- 6. Outputting results ---
print("\n--- Backtesting Results ---")
# Check that df_bt is not empty before accessing index
if not df_bt.empty:
    print(f"Period: {df_bt.index[0]} - {df_bt.index[-1]}")
print(f"Initial capital: {cerebro.broker.startingcash:.2f}")
print(f"Final capital: {cerebro.broker.getvalue():.2f}")
total_return = (cerebro.broker.getvalue() / cerebro.broker.startingcash - 1) * 100
print(f"Total return: {total_return:.2f}%")

# Use safe .get() access everywhere
trade_analysis = strat.analyzers.tradeanalyzer.get_analysis() if hasattr(strat.analyzers, 'tradeanalyzer') else {}
sharpe_analysis = strat.analyzers.sharpe.get_analysis() if hasattr(strat.analyzers, 'sharpe') else {}
drawdown_analysis = strat.analyzers.drawdown.get_analysis() if hasattr(strat.analyzers, 'drawdown') else {}
returns_analysis = strat.analyzers.returns.get_analysis() if hasattr(strat.analyzers, 'returns') else {}
sqn_analysis = strat.analyzers.sqn.get_analysis() if hasattr(strat.analyzers, 'sqn') else {}

print("\n--- Trade Analysis ---")
closed_trades = trade_analysis.get('total', {}).get('closed', 0)
if closed_trades > 0:
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    lost_trades = trade_analysis.get('lost', {}).get('total', 0)
    print(f"Total closed trades: {closed_trades}")
    print(f"Winning trades: {won_trades}")
    print(f"Losing trades: {lost_trades}")
    win_rate = won_trades / closed_trades * 100 if closed_trades > 0 else 0.0
    print(f"Win Rate: {win_rate:.2f}%")
    avg_pnl_net = trade_analysis.get('pnl', {}).get('net', {}).get('average', 0.0)
    print(f"Average PnL per trade (Net): {avg_pnl_net:.3f}")
    avg_win = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0.0)
    avg_loss = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0.0)
    if won_trades > 0: print(f"Average win: {avg_win:.3f}")
    if lost_trades > 0: print(f"Average loss: {avg_loss:.3f}")
    gross_profit = trade_analysis.get('pnl', {}).get('gross', {}).get('total', 0.0)
    gross_loss = abs(trade_analysis.get('pnl', {}).get('gross', {}).get('lost', 0.0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    print(f"Profit Factor (Gross): {profit_factor:.2f}")
else: print("No trades were made or insufficient data for analysis.")

print("\n--- Risk and Return Analysis ---")
sharpe_ratio = sharpe_analysis.get('sharperatio')
if sharpe_ratio is not None: print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
else: print("Failed to calculate Sharpe Ratio.")
max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0.0); max_dd_money = drawdown_analysis.get('max', {}).get('moneydown', 0.0)
print(f"Maximum drawdown: {max_dd:.2f}%"); print(f"Max. $ drawdown: {max_dd_money:.2f}")
rtot = returns_analysis.get('rtot') # Use get
if rtot is not None: print(f"Return over period (Total return): {rtot * 100:.2f}%")
else: print("Return over period (Total return): 0.00% (no data)")
sqn_val = sqn_analysis.get('sqn')
if sqn_val is not None: print(f"System Quality Number (SQN): {sqn_val:.2f}")

# --- 7. Chart ---
try:
    print("\nPlotting chart...")
    figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volup='#43A047', voldown='#E53935', dpi=100)[0][0]
    figure.savefig('backtest_chart_1h.png')
    print("Chart saved to 'backtest_chart_1h.png'")
except IndexError: print("Failed to plot chart (possibly no data).")
except Exception as e_plot: print(f"Failed to plot chart: {e_plot}")