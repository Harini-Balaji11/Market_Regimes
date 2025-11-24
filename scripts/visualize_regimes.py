"""
Regime Visualization Script

Creates comprehensive visualizations of market regimes:
- Price charts with regime overlays
- Regime transitions over time
- Model comparison (HMM vs K-Means vs GMM)
- Regime statistics visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.clustering import RegimeDetector
from models.hmm_model import HMMRegimeDetector

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_features(ticker):
    """Load features for a ticker"""
    features_file = Path(f'data/processed/{ticker}_features.csv')
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    features = pd.read_csv(features_file, index_col=0, parse_dates=True)
    print(f"  Loaded {len(features)} rows, {len(features.columns)} features")
    
    return features

def get_regime_labels(features, model_type='all'):
    """
    Get regime labels for all models
    
    Args:
        features: DataFrame with features
        model_type: 'all', 'kmeans', 'gmm', or 'hmm'
    
    Returns:
        Dictionary with model names and labels
    """
    results = {}
    
    # Select features (exclude price)
    feature_cols = [col for col in features.columns if col != 'close_price']
    X = features[feature_cols].dropna()
    prices = features.loc[X.index, 'close_price']
    
    print(f"\nFitting models on {len(X)} valid observations...")
    
    # K-Means
    if model_type in ['all', 'kmeans']:
        print("  Fitting K-Means...")
        detector = RegimeDetector(n_regimes=4, random_state=42)
        kmeans_labels = detector.fit_kmeans(X)
        results['K-Means'] = pd.Series(kmeans_labels, index=X.index)
    
    # GMM
    if model_type in ['all', 'gmm']:
        print("  Fitting GMM...")
        detector = RegimeDetector(n_regimes=4, random_state=42)
        gmm_labels, _ = detector.fit_gmm(X)
        results['GMM'] = pd.Series(gmm_labels, index=X.index)
    
    # HMM
    if model_type in ['all', 'hmm']:
        print("  Fitting HMM...")
        hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
        # HMM fit expects DataFrame, not array
        hmm_labels, _ = hmm_detector.fit(X)
        # Ensure labels are Series with proper index
        if isinstance(hmm_labels, np.ndarray):
            results['HMM'] = pd.Series(hmm_labels, index=X.index)
        else:
            results['HMM'] = hmm_labels
    
    return results, prices

def plot_price_with_regimes(prices, regime_labels, model_name, ticker, save_path):
    """
    Plot price chart with regime overlays
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(prices.index, prices.values, 'k-', linewidth=1.5, label='Price', alpha=0.8)
    
    # Color regions by regime
    colors = ['green', 'blue', 'orange', 'red']
    regime_names = ['Bull', 'Moderate Growth', 'Sideways', 'Bear']
    
    # Create shaded regions for each regime
    prev_date = prices.index[0]
    prev_regime = regime_labels.iloc[0]
    
    for date, regime in regime_labels.items():
        if regime != prev_regime or date == prices.index[-1]:
            # Shade the region
            ax.axvspan(prev_date, date, alpha=0.2, 
                      color=colors[int(prev_regime) % len(colors)],
                      label=f'Regime {int(prev_regime)}' if prev_date == prices.index[0] else '')
            prev_date = date
            prev_regime = regime
    
    # Final region
    ax.axvspan(prev_date, prices.index[-1], alpha=0.2,
              color=colors[int(prev_regime) % len(colors)])
    
    ax.set_title(f'{ticker} - Price with {model_name} Regimes', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path.name}")

def plot_regime_timeline(regime_labels, model_name, ticker, save_path):
    """
    Plot regime labels as a timeline
    """
    fig, ax = plt.subplots(figsize=(16, 3))
    
    # Plot regime timeline
    colors = ['green', 'blue', 'orange', 'red']
    
    for i, (date, regime) in enumerate(regime_labels.items()):
        ax.barh(0, 1, left=i, height=1, color=colors[int(regime) % len(colors)], 
               alpha=0.7, edgecolor='white', linewidth=0.5)
    
    ax.set_xlim(0, len(regime_labels))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(0, len(regime_labels), len(regime_labels) // 10))
    ax.set_xticklabels([regime_labels.index[i].strftime('%Y') 
                        for i in range(0, len(regime_labels), len(regime_labels) // 10)])
    ax.set_title(f'{ticker} - {model_name} Regime Timeline', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=f'Regime {i}') 
                      for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path.name}")

def plot_model_comparison(regime_dict, prices, ticker, save_path):
    """
    Compare all models side-by-side
    """
    n_models = len(regime_dict)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(16, 4 * (n_models + 1)), sharex=True)
    
    # Plot price on top
    ax = axes[0]
    ax.plot(prices.index, prices.values, 'k-', linewidth=1.5)
    ax.set_title(f'{ticker} - Price Chart', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot each model
    colors = ['green', 'blue', 'orange', 'red']
    
    for idx, (model_name, labels) in enumerate(regime_dict.items(), 1):
        ax = axes[idx]
        
        # Plot regime timeline
        for i, (date, regime) in enumerate(labels.items()):
            ax.barh(0, 1, left=i, height=1, 
                   color=colors[int(regime) % len(colors)], 
                   alpha=0.7, edgecolor='white', linewidth=0.5)
        
        ax.set_xlim(0, len(labels))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(model_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    axes[-1].set_xlabel('Time', fontsize=12)
    axes[-1].set_xticks(range(0, len(list(regime_dict.values())[0]), 
                              len(list(regime_dict.values())[0]) // 10))
    axes[-1].set_xticklabels([list(regime_dict.values())[0].index[i].strftime('%Y') 
                              for i in range(0, len(list(regime_dict.values())[0]), 
                                            len(list(regime_dict.values())[0]) // 10)])
    
    fig.suptitle(f'{ticker} - Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path.name}")

def plot_regime_statistics(regime_labels, prices, model_name, ticker, save_path):
    """
    Plot statistics for each regime
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    aligned_returns = returns.reindex(regime_labels.index).dropna()
    aligned_labels = regime_labels.reindex(aligned_returns.index)
    
    # Prepare data for plotting
    regime_data = []
    for regime in range(4):
        mask = aligned_labels == regime
        if mask.sum() > 0:
            regime_returns = aligned_returns[mask]
            regime_data.append({
                'Regime': regime,
                'Mean Return': regime_returns.mean() * 252,
                'Volatility': regime_returns.std() * np.sqrt(252),
                'Count': mask.sum()
            })
    
    stats_df = pd.DataFrame(regime_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Annual returns
    ax = axes[0, 0]
    bars = ax.bar(stats_df['Regime'].astype(str), stats_df['Mean Return'] * 100, 
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Annual Returns by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Return (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 2. Volatility
    ax = axes[0, 1]
    bars = ax.bar(stats_df['Regime'].astype(str), stats_df['Volatility'] * 100,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Annual Volatility by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Sharpe ratio
    ax = axes[1, 0]
    sharpe_ratios = (stats_df['Mean Return'] / stats_df['Volatility']).fillna(0)
    bars = ax.bar(stats_df['Regime'].astype(str), sharpe_ratios,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 4. Frequency
    ax = axes[1, 1]
    freq_pct = (stats_df['Count'] / stats_df['Count'].sum() * 100)
    bars = ax.bar(stats_df['Regime'].astype(str), freq_pct,
                  color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax.set_title('Regime Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.set_xlabel('Regime', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'{ticker} - {model_name} Regime Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path.name}")

def create_all_visualizations(ticker):
    """Create all visualizations for a ticker"""
    print(f"\n{'='*70}")
    print(f"CREATING VISUALIZATIONS FOR {ticker}")
    print('='*70)
    
    # Create output directory
    output_dir = Path('figures/regimes')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print(f"\nLoading features...")
    features = load_features(ticker)
    
    # Get regime labels from all models
    regime_dict, prices = get_regime_labels(features, model_type='all')
    
    # Create individual model visualizations
    print(f"\nCreating individual model visualizations...")
    for model_name, labels in regime_dict.items():
        model_name_clean = model_name.lower().replace('-', '_')
        
        # Price with regimes
        save_path = output_dir / f'{ticker}_{model_name_clean}_price_regimes.png'
        plot_price_with_regimes(prices, labels, model_name, ticker, save_path)
        
        # Timeline
        save_path = output_dir / f'{ticker}_{model_name_clean}_timeline.png'
        plot_regime_timeline(labels, model_name, ticker, save_path)
        
        # Statistics
        save_path = output_dir / f'{ticker}_{model_name_clean}_statistics.png'
        plot_regime_statistics(labels, prices, model_name, ticker, save_path)
    
    # Model comparison
    print(f"\nCreating model comparison...")
    save_path = output_dir / f'{ticker}_model_comparison.png'
    plot_model_comparison(regime_dict, prices, ticker, save_path)
    
    print(f"\n{'='*70}")
    print(f"VISUALIZATION COMPLETE FOR {ticker}")
    print(f"{'='*70}")
    print(f"\nAll figures saved to: {output_dir}")

def main():
    """Main function"""
    print("="*70)
    print("REGIME VISUALIZATION")
    print("="*70)
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    for ticker in tickers:
        try:
            create_all_visualizations(ticker)
        except FileNotFoundError as e:
            print(f"\n[SKIP] Skipping {ticker}: {e}")
        except Exception as e:
            print(f"\n[ERROR] Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nCheck the figures directory: figures/regimes/")

if __name__ == "__main__":
    main()

