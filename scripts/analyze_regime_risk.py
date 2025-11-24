"""
Regime-Aware Risk Analysis

Compute and visualize risk metrics per regime
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from risk_analytics.metrics import RiskMetrics, RegimeRiskAnalyzer
from models.hmm_model import HMMRegimeDetector
from models.clustering import RegimeDetector

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

def analyze_regime_risk(ticker='SPY'):
    """Analyze risk metrics per regime"""
    print("="*70)
    print(f"REGIME-AWARE RISK ANALYSIS - {ticker}")
    print("="*70)
    
    # Load features
    features = pd.read_csv(
        f'data/processed/{ticker}_features.csv',
        index_col=0,
        parse_dates=True
    )
    
    feature_cols = [col for col in features.columns if col != 'close_price']
    X = features[feature_cols].dropna()
    prices = features.loc[X.index, 'close_price']
    
    print(f"\nLoaded {len(X)} observations")
    
    # Get HMM regimes
    print("\nFitting HMM model...")
    hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
    regimes, _ = hmm_detector.fit(X)
    regimes = pd.Series(regimes, index=X.index)
    
    # Align data
    common_index = X.index.intersection(regimes.index)
    features_aligned = features.reindex(common_index)
    regimes_aligned = regimes.reindex(common_index)
    
    # Initialize risk analyzer
    risk_analyzer = RegimeRiskAnalyzer()
    
    print("\nComputing risk metrics per regime...")
    regime_risk = risk_analyzer.analyze_all_regimes(features_aligned, regimes_aligned)
    
    print("\n" + "="*70)
    print("REGIME-SPECIFIC RISK METRICS")
    print("="*70)
    # Select available columns (regime_name might not exist)
    cols_to_show = ['regime', 'frequency_pct', 'annual_return', 'annual_volatility', 
                    'sharpe_ratio', 'var_95', 'cvar_95']
    if 'regime_name' in regime_risk.columns:
        cols_to_show.insert(1, 'regime_name')
    if 'max_drawdown' in regime_risk.columns:
        cols_to_show.append('max_drawdown')
    
    print(regime_risk[cols_to_show].to_string(index=False))
    
    # Create visualizations
    output_dir = Path('figures/risk')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Risk-Return scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        regime_risk['annual_volatility'] * 100,
        regime_risk['annual_return'] * 100,
        s=regime_risk['frequency_pct'] * 50,  # Size by frequency
        c=regime_risk['regime'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    
    # Add labels
    for idx, row in regime_risk.iterrows():
        label_text = f"Regime {int(row['regime'])}"
        if 'regime_name' in row:
            label_text += f"\n{row['regime_name']}"
        ax.annotate(
            label_text,
            (row['annual_volatility'] * 100, row['annual_return'] * 100),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    ax.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{ticker} - Risk-Return Profile by Regime', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    save_path = output_dir / f'{ticker}_risk_return.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path.name}")
    
    # 2. Risk metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # VaR comparison
    ax = axes[0, 0]
    bars = ax.bar(regime_risk['regime'].astype(str), 
                  regime_risk['var_95'] * 100,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Value at Risk (95%) by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('VaR (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # CVaR comparison
    ax = axes[0, 1]
    bars = ax.bar(regime_risk['regime'].astype(str),
                  regime_risk['cvar_95'] * 100,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Conditional VaR (95%) by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('CVaR (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Sharpe ratio
    ax = axes[1, 0]
    bars = ax.bar(regime_risk['regime'].astype(str),
                  regime_risk['sharpe_ratio'],
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Max drawdown
    ax = axes[1, 1]
    bars = ax.bar(regime_risk['regime'].astype(str),
                  regime_risk['max_drawdown'] * 100,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Maximum Drawdown by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='top', fontsize=9)
    
    fig.suptitle(f'{ticker} - Risk Metrics Comparison by Regime',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{ticker}_risk_metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")
    
    # Save results
    regime_risk.to_csv(f'data/processed/{ticker}_regime_risk.csv', index=False)
    print(f"\nSaved detailed results: data/processed/{ticker}_regime_risk.csv")
    
    return regime_risk

def main():
    """Analyze risk for all tickers"""
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    for ticker in tickers:
        try:
            analyze_regime_risk(ticker)
            print("\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to analyze {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*70)
    print("REGIME RISK ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

