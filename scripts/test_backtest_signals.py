"""
Quick test script to debug backtest signal generation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.backtest import BullMarketStrategy
from models.hmm_model import HMMRegimeDetector

# Load data
ticker = 'SPY'
features = pd.read_csv(
    f'data/processed/{ticker}_features.csv',
    index_col=0,
    parse_dates=True
)

prices = features['close_price']
feature_cols = [col for col in features.columns if col != 'close_price']
X = features[feature_cols].dropna()

print("Fitting HMM...")
hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
regimes, _ = hmm_detector.fit(X)
regimes = pd.Series(regimes, index=X.index)

print(f"\nRegime distribution:")
print(regimes.value_counts().sort_index())

print(f"\nRegime range: {regimes.min()} to {regimes.max()}")

# Test strategy
common_index = prices.index.intersection(regimes.index)
prices_aligned = prices.reindex(common_index)
regimes_aligned = regimes.reindex(common_index)

print(f"\nAligned data: {len(prices_aligned)} prices, {len(regimes_aligned)} regimes")

strategy = BullMarketStrategy(bull_regimes=[0, 1, 2])  # Try broader range
signals = strategy.generate_signals(prices_aligned, regimes_aligned)

print(f"\nSignals generated:")
print(f"  Buy signals: {(signals == 1).sum()}")
print(f"  Sell signals: {(signals == -1).sum()}")
print(f"  Hold signals: {(signals == 0).sum()}")

if (signals == 1).sum() > 0:
    print(f"\nFirst few buy signals:")
    print(signals[signals == 1].head(10))
else:
    print("\nNo buy signals! Checking regime transitions...")
    is_bull = regimes_aligned.isin([0, 1, 2])
    transitions = (is_bull != is_bull.shift(1))
    print(f"  Regime transitions: {transitions.sum()}")
    print(f"  Is bull (first 10): {is_bull.head(10).tolist()}")
    print(f"  Regime values (first 10): {regimes_aligned.head(10).tolist()}")

