"""
Backtesting Script

Run backtests on all tickers with different strategies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.backtest import (
    BacktestRunner, BullMarketStrategy, RegimeRotationStrategy, 
    MomentumRegimeStrategy
)
from models.hmm_model import HMMRegimeDetector
import pandas as pd
import numpy as np

def main():
    """Run backtests for all tickers"""
    print("="*70)
    print("BACKTESTING ENGINE - All Tickers")
    print("="*70)
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    all_results = []
    
    for ticker in tickers:
        try:
            print(f"\n{'='*70}")
            print(f"Processing {ticker}")
            print('='*70)
            
            # Load features
            features = pd.read_csv(
                f'data/processed/{ticker}_features.csv',
                index_col=0,
                parse_dates=True
            )
            
            prices = features['close_price']
            feature_cols = [col for col in features.columns if col != 'close_price']
            X = features[feature_cols].dropna()
            
            # Fit HMM for regimes
            print("  Fitting HMM model...")
            hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
            regimes, _ = hmm_detector.fit(X)
            regimes = pd.Series(regimes, index=X.index)
            
            # Align prices with regimes
            common_index = prices.index.intersection(regimes.index)
            prices_aligned = prices.reindex(common_index)
            regimes_aligned = regimes.reindex(common_index)
            
            # Test strategies
            strategies = [
                BullMarketStrategy(),
                RegimeRotationStrategy(),
                MomentumRegimeStrategy()
            ]
            
            from models.backtest import BacktestRunner
            output_dir = Path('figures/backtest')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for strategy in strategies:
                print(f"\n  Testing: {strategy.name}")
                
                runner = BacktestRunner(initial_capital=100000, commission=0.001)
                results = runner.run(prices_aligned, regimes_aligned, strategy, 
                                   features.reindex(common_index))
                
                # Save visualization
                save_path = output_dir / f'{ticker}_{strategy.name.replace(" ", "_")}_backtest.png'
                runner.plot_results(results, prices_aligned, regimes_aligned, 
                                   strategy.name, ticker, save_path)
                
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': strategy.name,
                    'Total Return': results['total_return']*100,
                    'Annual Return': results['annual_return']*100,
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown': results['max_drawdown']*100,
                    'N Trades': results['n_trades'],
                    'Win Rate': results['win_rate']*100
                })
                
                print(f"    Return: {results['total_return']*100:.2f}% | "
                      f"Sharpe: {results['sharpe_ratio']:.2f} | "
                      f"Trades: {results['n_trades']}")
        
        except Exception as e:
            print(f"  [ERROR] Failed to process {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("BACKTESTING SUMMARY")
    print('='*70)
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('data/processed/backtest_summary.csv', index=False)
    print(f"\nSummary saved to: data/processed/backtest_summary.csv")
    
    print(f"\n{'='*70}")
    print("BACKTESTING COMPLETE")
    print('='*70)

if __name__ == "__main__":
    main()

