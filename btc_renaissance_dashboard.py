# btc_renaissance_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

# --- Global Settings and Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    warnings.filterwarnings('ignore', 'A value is trying to be set on a copy of a slice from a DataFrame')

st.title("BTC Renaissance Dashboard")

# --- Global Parameters for Rolling Regression ---
window_size = 50 

# --- Phase 1: Fetch BTC data ---
st.info("Phase 1: Fetching BTC 4h data...")
try:
    df_raw = yf.download("BTC-USD", interval="4h", period="30d")
    
    df = df_raw[['Close']].copy() 
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
    
    st.success("✅ BTC data fetched")
except Exception as e:
    st.error(f"Error fetching BTC data: {e}")
    st.stop()

# --- Shared Function for Rolling Regression (Coefficient Extraction) ---
# This function is used by the rolling apply calls in Phase 2
def get_roll_coeff_single(window, coeff_index):
    # 'window' is a Series of 'Close' prices
    x_window = np.arange(len(window))
    y_window = window.values.flatten()
    
    # np.polyfit returns (slope, intercept)
    coefficients = np.polyfit(x_window, y_window, 1)
    
    # Return only the coefficient requested (0 for slope, 1 for intercept)
    return coefficients[coeff_index]

# --- Phase 2: Compute ROLLING regression slope and intercept (FINAL FIX) ---
st.info(f"Phase 2: Computing **ROLLING** regression line (Window: {window_size} bars)...")
try:
    # 1. Calculate Rolling Slope (Coeff Index 0)
    df['slope'] = df['Close'].rolling(window=window_size).apply(
        lambda x: get_roll_coeff_single(x, 0), # Get the first coefficient (slope)
        raw=False
    )

    # 2. Calculate Rolling Intercept (Coeff Index 1)
    df['intercept'] = df['Close'].rolling(window=window_size).apply(
        lambda x: get_roll_coeff_single(x, 1), # Get the second coefficient (intercept)
        raw=False
    )
    
    st.success(f"✅ Rolling slope and intercept computed over {window_size} bars.")
except Exception as e:
    st.error(f"Error computing rolling regression: {e}")
    st.stop()

# --- Phase 3: Compute Rolling Bands and Regression Line ---
st.info("Phase 3: Computing dynamic regression bands...")
try:
    # 1. Calculate the rolling regression line (y = mx + c)
    # The 'x' value for the line is the end of the window (window_size - 1)
    df['reg'] = df['intercept'] + df['slope'] * (window_size - 1)
    
    # 2. Calculate Rolling Standard Deviation (Volatility)
    df['std_dev'] = df['Close'].rolling(window=window_size).std()

    # 3. Apply the dynamic bands (Regression Line +/- 2 * std_dev)
    df['upper_band'] = df['reg'] + 2 * df['std_dev']
    df['lower_band'] = df['reg'] - 2 * df['std_dev']
    
    st.success("✅ Dynamic regression bands computed.")
except Exception as e:
    st.error(f"Error computing bands: {e}")
    st.stop()

# --- Phase 4: Data Cleaning ---
st.info("Phase 4: Cleaning data...")
try:
    # The first (window_size - 1) rows will be NaN due to rolling window. Drop them.
    df = df.dropna()

    if df.empty:
        st.error("🚨 DataFrame is empty after cleaning. Cannot proceed.")
        st.stop()
    
    st.success(f"✅ Data cleaned. {len(df)} rows remaining (excluding initial {window_size-1} bars).")
except Exception as e:
    st.error(f"Error in Phase 4 data cleaning: {e}")
    st.stop()

# --- Phase 5: Generate dummy buy/sell signals ---
st.info("Phase 5: Generating dummy buy/sell signals...")
try:
    close_vals = df['Close'].values.flatten()
    lower_vals = df['lower_band'].values.flatten()
    upper_vals = df['upper_band'].values.flatten()
    
    df['buy_signal'] = close_vals < lower_vals
    df['sell_signal'] = close_vals > upper_vals
    
    st.success("✅ Dummy buy/sell signals generated")
except Exception as e:
    st.error(f"Error generating signals: {e}")
    pass 

# --- Phase 6: Display BTC chart with signals (Guarded & FIXED) ---
st.subheader("BTC Price & Rolling Regression Bands")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['reg'], mode='lines', name='Regression', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['upper_band'], mode='lines', name='Upper Band', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['lower_band'], mode='lines', name='Lower Band', line=dict(color='orange', dash='dot')))

    if 'buy_signal' in df.columns: 
        buys = df[df['buy_signal']]
        sells = df[df['sell_signal']]
        
        fig.add_trace(go.Scatter(x=buys['Datetime'], y=buys['Close'], mode='markers',
                                  marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sells['Datetime'], y=sells['Close'], mode='markers',
                                  marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
    
    # FIX: Force the Y-axis range to focus on actual price data
    min_price = df['Close'].min()
    max_price = df['Close'].max()
    buffer = (max_price - min_price) * 0.02 
    
    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Price (USD)", 
        hovermode="x unified",
        yaxis=dict(
            range=[min_price - buffer, max_price + buffer] 
        )
    )
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error plotting chart: {e}")
    st.stop()

# --- Phase 7: Show historical and latest signals ---
st.subheader("Latest Signal & History")

if 'buy_signal' in df.columns:
    
    ### 7a. Signal History Table
    st.markdown("### Recent Signal History")
    
    # Filter the DataFrame for Buy or Sell signals
    signal_history_df = df[df['buy_signal'] | df['sell_signal']].copy()
    
    # Create a clean 'Signal' column
    signal_history_df['Signal'] = np.where(signal_history_df['buy_signal'], 'BUY 🟢', 'SELL 🔴')

    # Select and format the final columns (show the last 10)
    history_table = signal_history_df[['Datetime', 'Close', 'Signal']].tail(10)
    history_table['Datetime'] = history_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    history_table['Close'] = history_table['Close'].round(2)
    
    if not history_table.empty:
        st.dataframe(history_table, use_container_width=True, hide_index=True)
    else:
        st.info("No Buy or Sell signals generated in the visible history.")

    ### 7b. Latest Signal
    st.markdown("### Current Bar Signal")
    latest = df.iloc[-1]
    
    if latest['buy_signal'].item():
        st.success("🟢 Latest signal: BUY")
    elif latest['sell_signal'].item():
        st.error("🔴 Latest signal: SELL")
    else:
        st.info("⚪ Latest signal: HOLD")
else:
    st.warning("⚠️ Signal generation failed in Phase 5. Cannot display latest signal.")