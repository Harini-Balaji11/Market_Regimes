"""
DAY 2: Feature Engineering
==========================

This creates all features needed for regime detection:
- Returns and volatility
- Moving averages
- Technical indicators (RSI, MACD, Bollinger Bands)

"""

import pandas as pd
import numpy as np
import ta
from pathlib import Path
def load_price_data(csv_file):
    """
    Load price CSV exported by download_data.py and ensure numeric types.
    
    The downloaded files contain two metadata rows ("Ticker" and "Date") before
    the actual timeseries values, so we drop them and coerce numeric columns.
    """
    df = pd.read_csv(csv_file)
    
    if df.columns[0] != 'Date':
        df = df.rename(columns={df.columns[0]: 'Date'})
    
    metadata_labels = {'Ticker', 'Date'}
    df = df[~df['Date'].isin(metadata_labels)]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    numeric_cols = [col for col in df.columns if col != 'Date']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    return df

def compute_returns(prices):
    """
    Calculate log returns
    
    Log returns are better for statistical analysis because:
    - They are time-additive
    - Approximately normally distributed
    - Handle percentage changes properly
    """
    return np.log(prices / prices.shift(1))

def compute_rolling_volatility(returns, window=30):
    """
    Calculate annualized rolling volatility
    
    Args:
        returns: Daily log returns
        window: Rolling window size in days
    
    Returns:
        Annualized volatility series
    """
    # Daily volatility
    daily_vol = returns.rolling(window=window).std()
    
    # Annualize (252 trading days per year)
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol

def compute_moving_averages(prices, windows=[10, 50, 200]):
    """
    Calculate simple moving averages
    
    Args:
        prices: Price series
        windows: List of window sizes
    
    Returns:
        DataFrame with MA columns
    """
    ma_df = pd.DataFrame(index=prices.index)
    
    for window in windows:
        ma_df[f'ma_{window}'] = prices.rolling(window=window).mean()
        
        # Also compute price-to-MA ratio (useful feature)
        ma_df[f'price_to_ma_{window}'] = prices / ma_df[f'ma_{window}']
    
    return ma_df

def compute_momentum(prices, periods=[5, 10, 20]):
    """
    Calculate momentum (rate of change)
    
    Args:
        prices: Price series
        periods: Lookback periods
    
    Returns:
        DataFrame with momentum columns
    """
    mom_df = pd.DataFrame(index=prices.index)
    
    for period in periods:
        mom_df[f'momentum_{period}d'] = prices.pct_change(periods=period)
    
    return mom_df

def compute_technical_indicators(ohlcv_data):
    """
    Compute technical indicators using TA library
    
    Args:
        ohlcv_data: DataFrame with Open, High, Low, Close, Volume
    
    Returns:
        DataFrame with technical indicators
    """
    indicators = pd.DataFrame(index=ohlcv_data.index)
    
    close = ohlcv_data['Close']
    high = ohlcv_data['High']
    low = ohlcv_data['Low']
    
    # RSI - Relative Strength Index (momentum oscillator)
    # Values: 0-100, >70 overbought, <30 oversold
    indicators['rsi'] = ta.momentum.RSIIndicator(
        close=close, 
        window=14
    ).rsi()
    
    # MACD - Moving Average Convergence Divergence
    macd = ta.trend.MACD(close=close)
    indicators['macd'] = macd.macd()
    indicators['macd_signal'] = macd.macd_signal()
    indicators['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands - Volatility indicator
    bollinger = ta.volatility.BollingerBands(close=close, window=20)
    indicators['bb_high'] = bollinger.bollinger_hband()
    indicators['bb_low'] = bollinger.bollinger_lband()
    indicators['bb_width'] = bollinger.bollinger_wband()
    
    # ATR - Average True Range (volatility measure)
    indicators['atr'] = ta.volatility.AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14
    ).average_true_range()
    
    # Volume indicators (if volume exists)
    if 'Volume' in ohlcv_data.columns:
        indicators['volume_sma'] = ohlcv_data['Volume'].rolling(20).mean()
        indicators['volume_ratio'] = ohlcv_data['Volume'] / indicators['volume_sma']
    
    return indicators

def compute_drawdown(prices):
    """
    Calculate drawdown from peak
    
    Drawdown shows how far price has fallen from its highest point
    """
    dd_df = pd.DataFrame(index=prices.index)
    
    # Cumulative returns
    cum_returns = (1 + prices.pct_change()).cumprod()
    
    # Running maximum (peak)
    running_max = cum_returns.expanding().max()
    
    # Drawdown as percentage
    dd_df['drawdown'] = (cum_returns - running_max) / running_max
    
    return dd_df

def create_features_for_ticker(ticker, data_dir='data/raw'):
    """
    Create complete feature set for one ticker
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY')
        data_dir: Directory containing raw CSV files
    
    Returns:
        DataFrame with all computed features
    """
    print(f"\nProcessing {ticker}...")
    
    # Load raw data
    csv_file = Path(data_dir) / f"{ticker}.csv"
    
    if not csv_file.exists():
        print(f"  ✗ File not found: {csv_file}")
        return None
    
    df = load_price_data(csv_file)
    print(f"  Loaded {len(df)} rows from {csv_file}")
    
    # Initialize feature DataFrame
    features = pd.DataFrame(index=df.index)
    
    # Store close price (we'll need this later)
    features['close_price'] = df['Close']
    
    # 1. Returns
    features['returns'] = compute_returns(df['Close'])
    
    # 2. Volatility (multiple time windows)
    for window in [10, 30, 60, 90]:
        col_name = f'vol_{window}d'
        features[col_name] = compute_rolling_volatility(
            features['returns'], 
            window=window
        )
    
    # 3. Moving Averages
    ma_df = compute_moving_averages(df['Close'], windows=[10, 50, 200])
    features = features.join(ma_df)
    
    # 4. Momentum
    mom_df = compute_momentum(df['Close'], periods=[5, 10, 20])
    features = features.join(mom_df)
    
    # 5. Technical Indicators
    tech_indicators = compute_technical_indicators(df)
    features = features.join(tech_indicators)
    
    # 6. Drawdown
    dd_df = compute_drawdown(df['Close'])
    features = features.join(dd_df)
    
    # Count features
    n_features = len(features.columns)
    n_valid = features.notna().all(axis=1).sum()
    
    print(f"  ✓ Created {n_features} features")
    print(f"  Valid rows (no NaN): {n_valid} / {len(features)}")
    
    return features

def save_features(features, ticker, output_dir='data/processed'):
    """Save features to CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{ticker}_features.csv"
    features.to_csv(output_file)
    
    print(f"  ✓ Saved to {output_file}")

def main():
    """Process all tickers"""
    print("="*70)
    print("FEATURE ENGINEERING - Creating Features for All Tickers")
    print("="*70)
    
    # List of tickers to process
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    results = {}
    
    for ticker in tickers:
        try:
            # Create features
            features = create_features_for_ticker(ticker)
            
            if features is not None:
                # Save to processed directory
                save_features(features, ticker)
                results[ticker] = 'Success'
            else:
                results[ticker] = 'Failed - file not found'
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[ticker] = f'Failed - {str(e)[:50]}'
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for ticker, status in results.items():
        symbol = "✓" if status == "Success" else "✗"
        print(f"{symbol} {ticker}: {status}")
    
    print("\n✅ Feature engineering complete!")
    print("\nNext steps:")
    print("1. Review features: data/processed/SPY_features.csv")
    print("2. Git commit: git add . && git commit -m 'Add feature engineering'")
    print("3. Move to Day 3: Model training")

if __name__ == "__main__":
    main()