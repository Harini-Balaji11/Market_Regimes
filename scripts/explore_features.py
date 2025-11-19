"""
Day 2: Quick Data Exploration
==============================

Run this after feature engineering to see what you created

Usage: python explore_features.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

def explore_ticker(ticker):
    """Explore features for one ticker"""
    
    print(f"\n{'='*70}")
    print(f"EXPLORING {ticker}")
    print('='*70)
    
    # Load features
    features_file = f'data/processed/{ticker}_features.csv'
    
    if not Path(features_file).exists():
        print(f"✗ File not found: {features_file}")
        return
    
    df = pd.read_csv(features_file, index_col=0, parse_dates=True)
    
    # Basic info
    print(f"\n Dataset Info:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
    
    # Features list
    print(f"\n Features Created ({len(df.columns)}):")
    print(f"  {df.columns.tolist()}")
    
    # Missing values
    print(f"\n Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values in feature columns")
    
    # Basic statistics
    print(f"\n Key Statistics:")
    print(f"  Mean daily return: {df['returns'].mean():.5f} ({df['returns'].mean()*252*100:.2f}% annualized)")
    print(f"  Daily volatility: {df['returns'].std():.5f} ({df['returns'].std()*np.sqrt(252)*100:.2f}% annualized)")
    print(f"  Mean RSI: {df['rsi'].mean():.2f}")
    print(f"  Current price: ${df['close_price'].iloc[-1]:.2f}")
    print(f"  Max drawdown: {df['drawdown'].min():.2%}")
    
    # Create visualization
    create_visualization(df, ticker)
    
    return df

def create_visualization(df, ticker):
    """Create summary plots"""
    
    print(f"\n Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} - Feature Overview', fontsize=16, fontweight='bold')
    
    # 1. Price history
    ax = axes[0, 0]
    ax.plot(df.index, df['close_price'], linewidth=1.5)
    ax.set_title('Price History')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    
    # 2. Returns distribution
    ax = axes[0, 1]
    df['returns'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')
    
    # 3. Rolling volatility
    ax = axes[1, 0]
    ax.plot(df.index, df['vol_30d'], label='30-day', linewidth=1.5)
    ax.plot(df.index, df['vol_60d'], label='60-day', linewidth=1.5, alpha=0.7)
    ax.set_title('Rolling Volatility')
    ax.set_ylabel('Annualized Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. RSI
    ax = axes[1, 1]
    ax.plot(df.index, df['rsi'], linewidth=1, color='purple')
    ax.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
    ax.set_title('RSI (Relative Strength Index)')
    ax.set_ylabel('RSI')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. MACD
    ax = axes[2, 0]
    ax.plot(df.index, df['macd'], label='MACD', linewidth=1)
    ax.plot(df.index, df['macd_signal'], label='Signal', linewidth=1)
    ax.fill_between(df.index, df['macd_diff'], 0, alpha=0.3, label='Histogram')
    ax.set_title('MACD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Drawdown
    ax = axes[2, 1]
    ax.fill_between(df.index, df['drawdown'] * 100, 0, 
                     color='red', alpha=0.3)
    ax.set_title('Drawdown from Peak')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('figures/eda')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{ticker}_features_overview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    
    plt.show()

def main():
    """Explore all tickers"""
    
    print("="*70)
    print("FEATURE EXPLORATION")
    print("="*70)
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    for ticker in tickers:
        try:
            df = explore_ticker(ticker)
        except Exception as e:
            print(f"\n✗ Error exploring {ticker}: {e}")
    
    print("\n" + "="*70)
    print("✅ EXPLORATION COMPLETE")
    print("="*70)
    print("\nGenerated figures:")
    print("  - figures/eda/SPY_features_overview.png")
    print("  - figures/eda/QQQ_features_overview.png")
    print("  - figures/eda/AAPL_features_overview.png")
    print("  - figures/eda/TSLA_features_overview.png")

if __name__ == "__main__":
    main()