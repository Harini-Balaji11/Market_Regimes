"""
Model Comparison Script

Compare all regime detection models and recommend best approach
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.clustering import RegimeDetector
from models.hmm_model import HMMRegimeDetector

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

def compare_models(ticker='SPY'):
    """Compare all regime detection models"""
    print("="*70)
    print(f"MODEL COMPARISON - {ticker}")
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
    returns = features.loc[X.index, 'returns'].dropna()
    
    print(f"\nLoaded {len(X)} observations with {len(X.columns)} features")
    
    # Fit all models
    print("\nFitting models...")
    
    # K-Means
    print("  K-Means...")
    kmeans_detector = RegimeDetector(n_regimes=4, random_state=42)
    kmeans_labels = kmeans_detector.fit_kmeans(X)
    kmeans_labels = pd.Series(kmeans_labels, index=X.index)
    
    # GMM
    print("  GMM...")
    gmm_detector = RegimeDetector(n_regimes=4, random_state=42)
    gmm_labels, gmm_probs = gmm_detector.fit_gmm(X)
    gmm_labels = pd.Series(gmm_labels, index=X.index)
    
    # HMM
    print("  HMM...")
    hmm_detector = HMMRegimeDetector(n_regimes=4, random_state=42)
    hmm_labels, trans_matrix = hmm_detector.fit(X)
    hmm_labels = pd.Series(hmm_labels, index=X.index)
    
    # Align all labels
    common_index = X.index
    kmeans_aligned = kmeans_labels.reindex(common_index)
    gmm_aligned = gmm_labels.reindex(common_index)
    hmm_aligned = hmm_labels.reindex(common_index)
    
    # Analyze each model
    results = {}
    
    for model_name, labels in [('K-Means', kmeans_aligned), 
                               ('GMM', gmm_aligned), 
                               ('HMM', hmm_aligned)]:
        print(f"\n{'-'*70}")
        print(f"Analyzing {model_name}")
        print('-'*70)
        
        # Regime statistics
        analyzer = RegimeDetector(n_regimes=4)
        stats = analyzer.analyze_regimes(X, labels, prices)
        
        # Calculate regime coherence (how often regime stays same)
        regime_changes = (labels != labels.shift(1)).sum()
        coherence = 1 - (regime_changes / len(labels))
        
        # Calculate average regime duration
        durations = []
        current_regime = labels.iloc[0]
        duration = 1
        for i in range(1, len(labels)):
            if labels.iloc[i] == current_regime:
                duration += 1
            else:
                durations.append(duration)
                current_regime = labels.iloc[i]
                duration = 1
        durations.append(duration)
        avg_duration = np.mean(durations)
        
        # Model agreement (if comparing with others)
        results[model_name] = {
            'regime_stats': stats,
            'n_regime_changes': regime_changes,
            'coherence': coherence,
            'avg_duration_days': avg_duration,
            'labels': labels
        }
        
        print(f"\nRegime Statistics:")
        print(stats[['regime_name', 'frequency_pct', 'annual_return', 
                    'annual_volatility', 'sharpe_ratio']].to_string(index=False))
        print(f"\nCoherence: {coherence:.3f} (higher = more persistent)")
        print(f"Average regime duration: {avg_duration:.1f} days")
        print(f"Number of regime changes: {regime_changes}")
    
    # Model agreement analysis
    print(f"\n{'='*70}")
    print("MODEL AGREEMENT ANALYSIS")
    print('='*70)
    
    # Pairwise agreement
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    models = ['K-Means', 'GMM', 'HMM']
    label_dict = {
        'K-Means': kmeans_aligned,
        'GMM': gmm_aligned,
        'HMM': hmm_aligned
    }
    
    agreement_matrix = np.zeros((3, 3))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            labels1 = label_dict[model1].dropna()
            labels2 = label_dict[model2].dropna()
            common_idx = labels1.index.intersection(labels2.index)
            if len(common_idx) > 0:
                ari = adjusted_rand_score(
                    labels1.reindex(common_idx).fillna(0),
                    labels2.reindex(common_idx).fillna(0)
                )
                agreement_matrix[i, j] = ari
    
    agreement_df = pd.DataFrame(
        agreement_matrix,
        index=models,
        columns=models
    )
    
    print("\nAdjusted Rand Index (1.0 = perfect agreement):")
    print(agreement_df.round(3).to_string())
    
    # Create comparison visualization
    output_dir = Path('figures/regimes')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Price chart
    ax = axes[0]
    ax.plot(prices.index, prices.values, 'k-', linewidth=1.5)
    ax.set_title(f'{ticker} - Price Chart', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Model timelines
    colors = ['green', 'blue', 'orange', 'red']
    
    for idx, (model_name, labels) in enumerate([('K-Means', kmeans_aligned),
                                                  ('GMM', gmm_aligned),
                                                  ('HMM', hmm_aligned)], 1):
        ax = axes[idx]
        
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
    fig.suptitle(f'{ticker} - Model Comparison Timeline', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{ticker}_model_comparison_timeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison visualization: {save_path}")
    
    # Summary recommendation
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATION")
    print('='*70)
    
    # Find model with highest coherence (most persistent regimes)
    best_coherence = max(results.items(), key=lambda x: x[1]['coherence'])
    print(f"\nHighest Regime Coherence: {best_coherence[0]} ({best_coherence[1]['coherence']:.3f})")
    print("  -> Best for identifying stable market periods")
    
    # Find model with best regime separation (highest Sharpe spread)
    sharpe_spreads = {}
    for model_name, result in results.items():
        stats = result['regime_stats']
        if 'sharpe_ratio' in stats.columns:
            spread = stats['sharpe_ratio'].max() - stats['sharpe_ratio'].min()
            sharpe_spreads[model_name] = spread
    
    if sharpe_spreads:
        best_separation = max(sharpe_spreads.items(), key=lambda x: x[1])
        print(f"\nBest Regime Separation: {best_separation[0]} (Sharpe spread: {best_separation[1]:.2f})")
        print("  -> Best for distinguishing high/low performing regimes")
    
    print(f"\nRecommendation: HMM typically best for regime-aware strategies")
    print("  - Captures temporal dependencies")
    print("  - Provides transition probabilities")
    print("  - Good for predicting regime changes")
    
    return results

def main():
    """Compare models for all tickers"""
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    all_results = {}
    
    for ticker in tickers:
        try:
            results = compare_models(ticker)
            all_results[ticker] = results
        except Exception as e:
            print(f"\n[ERROR] Failed to compare models for {ticker}: {e}")
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON COMPLETE")
    print('='*70)

if __name__ == "__main__":
    main()

