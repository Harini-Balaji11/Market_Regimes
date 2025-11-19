"""
Hidden Markov Model for Regime Detection

HMM captures temporal dependencies between market regimes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

class HMMRegimeDetector:
    """Detect market regimes using Hidden Markov Model"""
    
    def __init__(self, n_regimes=4, random_state=42, n_iter=100):
        """
        Initialize HMM detector
        
        Args:
            n_regimes: Number of hidden states (regimes)
            random_state: Random seed for reproducibility
            n_iter: Maximum EM iterations
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.n_iter = n_iter
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, features):
        """
        Fit HMM using Gaussian emissions
        
        Args:
            features: DataFrame of features
        
        Returns:
            Tuple of (labels, transition_matrix)
        """
        print(f"\nFitting HMM with {self.n_regimes} states...")
        
        # For HMM, focus on key features (returns and volatility)
        if 'returns' in features.columns and 'vol_30d' in features.columns:
            feature_subset = ['returns', 'vol_30d']
            print(f"  Using features: {feature_subset}")
            X = features[feature_subset].values
        else:
            print(f"  Using all {features.shape[1]} features")
            X = features.values
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )
        
        self.model.fit(X_scaled)
        
        # Predict most likely state sequence (Viterbi algorithm)
        labels = self.model.predict(X_scaled)
        
        # Get transition matrix
        transition_matrix = self.model.transmat_
        
        # Model diagnostics
        converged = self.model.monitor_.converged
        log_likelihood = self.model.score(X_scaled)
        
        print(f"  Converged: {converged}")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Iterations: {self.model.monitor_.iter}")
        
        return labels, transition_matrix
    
    def analyze_transitions(self, transition_matrix):
        """
        Analyze regime transition probabilities
        
        Args:
            transition_matrix: Matrix of transition probabilities
        
        Returns:
            DataFrame with formatted transition probabilities
        """
        trans_df = pd.DataFrame(
            transition_matrix,
            columns=[f'To_Regime_{i}' for i in range(self.n_regimes)],
            index=[f'From_Regime_{i}' for i in range(self.n_regimes)]
        )
        
        return trans_df
    
    def get_regime_persistence(self, transition_matrix):
        """
        Calculate how persistent each regime is (diagonal of transition matrix)
        
        Args:
            transition_matrix: Transition probability matrix
        
        Returns:
            Dictionary of persistence probabilities
        """
        persistence = {}
        
        for i in range(self.n_regimes):
            persistence[f'Regime_{i}'] = transition_matrix[i, i]
        
        return persistence

def main():
    """Test the HMM model"""
    print("="*70)
    print("REGIME DETECTION - Hidden Markov Model")
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
        
        print(f"  Loaded {len(features)} rows")
        
    except FileNotFoundError:
        print(f"  ERROR: File not found!")
        print(f"  Make sure you ran Day 2 feature engineering first")
        return
    
    # Select features
    feature_cols = ['returns', 'vol_30d', 'rsi', 'macd']
    X = features[feature_cols].dropna()
    prices = features.loc[X.index, 'close_price']
    
    print(f"\nUsing {len(X)} observations with features: {feature_cols}")
    
    # Initialize and fit HMM
    detector = HMMRegimeDetector(n_regimes=4, random_state=42, n_iter=100)
    
    labels, transition_matrix = detector.fit(X)
    
    # Regime distribution
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION")
    print("="*70)
    
    regime_counts = pd.Series(labels).value_counts().sort_index()
    print(regime_counts)
    print(f"\nTotal observations: {len(labels)}")
    
    # Transition matrix
    print("\n" + "="*70)
    print("TRANSITION PROBABILITY MATRIX")
    print("="*70)
    
    trans_df = detector.analyze_transitions(transition_matrix)
    print(trans_df.round(3))
    
    # Regime persistence
    print("\n" + "="*70)
    print("REGIME PERSISTENCE (Diagonal)")
    print("="*70)
    
    persistence = detector.get_regime_persistence(transition_matrix)
    for regime, prob in persistence.items():
        print(f"  {regime}: {prob:.3f} ({prob*100:.1f}% stay in same regime)")
    
    # Analyze regime characteristics
    from clustering import RegimeDetector
    
    analyzer = RegimeDetector(n_regimes=4)
    regime_stats = analyzer.analyze_regimes(X, labels, prices)
    
    print("\n" + "="*70)
    print("HMM REGIME STATISTICS")
    print("="*70)
    
    print(regime_stats[['regime_name', 'frequency_pct', 'annual_return',
                        'annual_volatility', 'sharpe_ratio']].to_string(index=False))
    
    # Save results
    regime_stats.to_csv(f'data/processed/{ticker}_hmm_regimes.csv', index=False)
    
    trans_df.to_csv(f'data/processed/{ticker}_hmm_transitions.csv')
    
    print(f"\n  Saved regime stats to data/processed/{ticker}_hmm_regimes.csv")
    print(f"  Saved transitions to data/processed/{ticker}_hmm_transitions.csv")
    
    print("\n" + "="*70)
    print("HMM COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Compare K-Means, GMM, and HMM results")
    print("  2. Choose best model (usually HMM)")
    print("  3. Move to Day 4: Risk Analytics")

if __name__ == "__main__":
    main()
