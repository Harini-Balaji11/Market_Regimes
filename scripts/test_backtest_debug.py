"""
Debug script to test backtest signal generation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.backtest import BacktestRunner, BullMarketStrategy, RegimeRotationStrategy, MomentumRegimeStrategy
from models.hmm_model import HMMRegimeDetector

def test_backtest():
    """Test backtest with debug output"""
    ticker = 'SPY'
    print(f"Testing backtest for {ticker}...")
    
    # Load features
    features = pd.read_csv(
        f'data/processed/{ticker}_features.csv',
        index_col=0,
        parse_dates=True
    )
    
    feature_cols = [col for col in features.columns if col != 'close_price']
    X = features[feature_cols].dropna()
    prices = features.loc[X.index, 'close_price']
    
    # Fit HMM
    print("\nFitting HMM...")
    hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
    regimes, _ = hmm_detector.fit(X)
    regimes = pd.Series(regimes, index=X.index)
    
    # Align
    common_index = prices.index.intersection(regimes.index)
    prices_aligned = prices.reindex(common_index)
    regimes_aligned = regimes.reindex(common_index)
    
    print(f"\nData aligned: {len(common_index)} observations")
    print(f"Regimes found: {sorted(regimes_aligned.unique())}")
    print(f"Regime distribution:")
    print(regimes_aligned.value_counts().sort_index())
    
    # Analyze regime returns
    returns = features.loc[common_index, 'returns'].dropna()
    common_idx2 = returns.index.intersection(regimes_aligned.index)
    returns_aligned = returns.reindex(common_idx2)
    regimes_aligned2 = regimes_aligned.reindex(common_idx2)
    
    print(f"\nRegime return analysis:")
    for regime in sorted(regimes_aligned2.unique()):
        regime_returns = returns_aligned[regimes_aligned2 == regime]
        mean_ret = regime_returns.mean() * 252
        print(f"  Regime {regime}: Mean annual return = {mean_ret*100:.2f}% (from {len(regime_returns)} observations)")
    
    # Test Bull Market Strategy
    print(f"\n{'='*70}")
    print("Testing Bull Market Strategy")
    print('='*70)
    
    strategy = BullMarketStrategy()
    signals = strategy.generate_signals(prices_aligned, regimes_aligned, features.reindex(common_index))
    
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    
    print(f"Buy signals: {n_buy}")
    print(f"Sell signals: {n_sell}")
    print(f"Bull regimes detected: {strategy.bull_regimes}")
    
    if n_buy > 0 or n_sell > 0:
        print("\nFirst few signals:")
        signal_dates = signals[signals != 0].head(10)
        for date, signal in signal_dates.items():
            regime = regimes_aligned.loc[date]
            price = prices_aligned.loc[date]
            action = "BUY" if signal == 1 else "SELL"
            print(f"  {date.date()}: {action} at ${price:.2f} (Regime {regime})")
        
        # Run backtest
        print("\nRunning backtest...")
        runner = BacktestRunner(initial_capital=100000, commission=0.001)
        results = runner.run(prices_aligned, regimes_aligned, strategy, features.reindex(common_index))
        
        print(f"\nResults:")
        print(f"  Total Return: {results['total_return']*100:.2f}%")
        print(f"  Annual Return: {results['annual_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"  Number of Trades: {results['n_trades']}")
        print(f"  Win Rate: {results['win_rate']*100:.2f}%")
    else:
        print("\n[WARNING] No signals generated!")
        print("Checking signal generation logic...")

if __name__ == "__main__":
    test_backtest()

