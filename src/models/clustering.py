"""
Clustering-based Regime Detection

Implements K-Means and Gaussian Mixture Models for regime detection
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

class RegimeDetector:
    """Detect market regimes using clustering methods"""
    
    def __init__(self, n_regimes=4, random_state=42):
        """
        Initialize detector
        
        Args:
            n_regimes: Number of regimes to identify
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.gmm_model = None
        
    def fit_kmeans(self, features):
        """
        Fit K-Means clustering model
        
        Args:
            features: DataFrame or array of features
        
        Returns:
            Array of cluster labels
        """
        print(f"\nFitting K-Means with {self.n_regimes} clusters...")
        
        # Standardize features (important for K-Means!)
        X_scaled = self.scaler.fit_transform(features)
        
        # Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=20,  # Multiple initializations for better results
            max_iter=300
        )
        
        labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Evaluate clustering quality
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        print(f"  Silhouette Score: {silhouette:.3f} (higher is better)")
        print(f"  Davies-Bouldin Score: {db_score:.3f} (lower is better)")
        
        return labels
    
    def fit_gmm(self, features):
        """
        Fit Gaussian Mixture Model
        
        GMM provides soft clustering (probabilistic regime assignment)
        
        Args:
            features: DataFrame or array of features
        
        Returns:
            Tuple of (labels, probabilities)
        """
        print(f"\nFitting GMM with {self.n_regimes} components...")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',  # Full covariance matrix
            random_state=self.random_state,
            n_init=10,
            max_iter=200
        )
        
        self.gmm_model.fit(X_scaled)
        
        # Get hard labels and soft probabilities
        labels = self.gmm_model.predict(X_scaled)
        probabilities = self.gmm_model.predict_proba(X_scaled)
        
        # Model selection metrics
        bic = self.gmm_model.bic(X_scaled)
        aic = self.gmm_model.aic(X_scaled)
        
        print(f"  BIC: {bic:.2f} (lower is better)")
        print(f"  AIC: {aic:.2f} (lower is better)")
        
        # Average confidence
        max_probs = probabilities.max(axis=1)
        avg_confidence = max_probs.mean()
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        return labels, probabilities
    
    def analyze_regimes(self, features, labels, prices=None):
        """
        Analyze characteristics of detected regimes
        
        Args:
            features: Feature DataFrame
            labels: Regime labels
            prices: Optional price series for analysis
        
        Returns:
            DataFrame with regime statistics
        """
        results = []
        
        for regime in range(self.n_regimes):
            mask = labels == regime
            regime_data = features[mask]
            
            stats = {
                'regime': regime,
                'n_samples': mask.sum(),
                'frequency_pct': mask.mean() * 100,
            }
            
            # Compute statistics for returns if available
            if 'returns' in features.columns:
                returns = regime_data['returns']
                stats.update({
                    'mean_return': returns.mean(),
                    'std_return': returns.std(),
                    'annual_return': returns.mean() * 252,
                    'annual_volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                })
            
            results.append(stats)
        
        regime_df = pd.DataFrame(results)
        
        # Assign intuitive names
        regime_df['regime_name'] = self._name_regimes(regime_df)
        
        return regime_df
    
    def _name_regimes(self, regime_df):
        """Assign intuitive names based on return/volatility characteristics"""
        names = []
        
        for _, row in regime_df.iterrows():
            if 'annual_return' in row and 'annual_volatility' in row:
                ret = row['annual_return']
                vol = row['annual_volatility']
                
                if ret > 0.10 and vol < 0.20:
                    name = "Bull Market"
                elif ret > 0.05 and vol >= 0.25:
                    name = "High Volatility Rally"
                elif ret < 0 and vol >= 0.25:
                    name = "Bear Market"
                elif ret < 0 and vol < 0.20:
                    name = "Slow Decline"
                else:
                    name = "Sideways Market"
            else:
                name = f"Regime {int(row['regime'])}"
            
            names.append(name)
        
        return names

def main():
    """Test the clustering models"""
    print("="*70)
    print("REGIME DETECTION - Clustering Models")
    print("="*70)
    
    # Load features
    ticker = 'SPY'
    print(f"\nLoading features for {ticker}...")
    
    try:
        features = pd.read_csv(
            f'data/processed/{ticker}_features.csv',
            index_col=0,
            parse_dates=True
        )
        
        print(f"  Loaded {len(features)} rows, {len(features.columns)} features")
        
    except FileNotFoundError:
        print(f"  ERROR: File not found!")
        print(f"  Make sure you ran Day 2 feature engineering first")
        return
    
    # Select features for clustering (exclude price)
    feature_cols = [col for col in features.columns if col != 'close_price']
    X = features[feature_cols].dropna()
    prices = features.loc[X.index, 'close_price']
    
    print(f"\nUsing {len(X)} valid observations with {len(X.columns)} features")
    
    # Initialize detector
    detector = RegimeDetector(n_regimes=4, random_state=42)
    
    # Fit K-Means
    print("\n" + "="*70)
    print("K-MEANS CLUSTERING")
    print("="*70)
    
    kmeans_labels = detector.fit_kmeans(X)
    
    print("\nRegime Distribution:")
    print(pd.Series(kmeans_labels).value_counts().sort_index())
    
    kmeans_stats = detector.analyze_regimes(X, kmeans_labels, prices)
    print("\nK-Means Regime Statistics:")
    print(kmeans_stats[['regime_name', 'frequency_pct', 'annual_return', 
                        'annual_volatility', 'sharpe_ratio']].to_string(index=False))
    
    # Save K-Means results
    kmeans_stats.to_csv(f'data/processed/{ticker}_kmeans_regimes.csv', index=False)
    print(f"\n  Saved to data/processed/{ticker}_kmeans_regimes.csv")
    
    # Fit GMM
    print("\n" + "="*70)
    print("GAUSSIAN MIXTURE MODEL")
    print("="*70)
    
    gmm_labels, gmm_probs = detector.fit_gmm(X)
    
    print("\nRegime Distribution:")
    print(pd.Series(gmm_labels).value_counts().sort_index())
    
    gmm_stats = detector.analyze_regimes(X, gmm_labels, prices)
    print("\nGMM Regime Statistics:")
    print(gmm_stats[['regime_name', 'frequency_pct', 'annual_return',
                     'annual_volatility', 'sharpe_ratio']].to_string(index=False))
    
    # Save GMM results
    gmm_stats.to_csv(f'data/processed/{ticker}_gmm_regimes.csv', index=False)
    print(f"\n  Saved to data/processed/{ticker}_gmm_regimes.csv")
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review regime statistics above")
    print("  2. Continue to HMM model")
    print("  3. Git commit your changes")

if __name__ == "__main__":
    main()
