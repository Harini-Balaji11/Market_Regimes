"""
Risk Analytics Module

Comprehensive risk metrics for regime analysis
"""

import numpy as np
import pandas as pd
from scipy import stats

class RiskMetrics:
    """Calculate comprehensive risk metrics"""
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize risk calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def value_at_risk(self, returns, confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        
        VaR represents the maximum expected loss at a given confidence level
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level (default 95%)
        
        Returns:
            VaR as negative number (representing loss)
        """
        return np.percentile(returns.dropna(), (1 - confidence) * 100)
    
    def conditional_var(self, returns, confidence=0.95):
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall)
        
        CVaR is the expected loss given that loss exceeds VaR
        More conservative than VaR
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level
        
        Returns:
            CVaR (average of losses beyond VaR)
        """
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()
    
    def sharpe_ratio(self, returns, periods_per_year=252):
        """
        Calculate annualized Sharpe ratio
        
        Measures risk-adjusted returns
        
        Args:
            returns: Series of daily returns
            periods_per_year: Trading days per year (default 252)
        
        Returns:
            Sharpe ratio (higher is better)
        """
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - self.risk_free_rate) / std_return
    
    def sortino_ratio(self, returns, periods_per_year=252):
        """
        Calculate Sortino ratio
        
        Like Sharpe but only penalizes downside volatility
        
        Args:
            returns: Series of daily returns
            periods_per_year: Trading days per year
        
        Returns:
            Sortino ratio
        """
        mean_return = returns.mean() * periods_per_year
        
        # Only negative returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_std == 0:
            return 0.0
        
        return (mean_return - self.risk_free_rate) / downside_std
    
    def max_drawdown(self, prices):
        """
        Calculate maximum drawdown
        
        Maximum peak-to-trough decline
        
        Args:
            prices: Series of prices
        
        Returns:
            Dictionary with max_drawdown, duration, current_drawdown
        """
        cum_returns = (1 + prices.pct_change()).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        
        # Calculate duration
        dd_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
            else:
                dd_duration = max(dd_duration, current_duration)
                current_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'duration_days': dd_duration,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def compute_all_metrics(self, returns, prices=None):
        """
        Compute comprehensive risk metrics
        
        Args:
            returns: Series of returns
            prices: Optional price series for drawdown
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'n_observations': len(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            
            # Risk metrics
            'var_95': self.value_at_risk(returns, 0.95),
            'cvar_95': self.conditional_var(returns, 0.95),
            'var_99': self.value_at_risk(returns, 0.99),
            'cvar_99': self.conditional_var(returns, 0.99),
            
            # Risk-adjusted returns
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            
            # Distribution
            'skewness': stats.skew(returns.dropna()),
            'kurtosis': stats.kurtosis(returns.dropna()),
            
            # Percentiles
            'percentile_5': np.percentile(returns.dropna(), 5),
            'percentile_95': np.percentile(returns.dropna(), 95),
        }
        
        # Add drawdown if prices provided
        if prices is not None:
            dd_metrics = self.max_drawdown(prices)
            metrics.update(dd_metrics)
        
        return metrics

class RegimeRiskAnalyzer:
    """Analyze risk metrics per regime"""
    
    def __init__(self):
        self.risk_calc = RiskMetrics()
    
    def analyze_all_regimes(self, features, labels):
        """
        Compute metrics for all regimes
        
        Args:
            features: DataFrame with returns and prices
            labels: Regime labels
        
        Returns:
            DataFrame with metrics per regime
        """
        results = []
        
        unique_regimes = np.unique(labels)
        
        for regime in unique_regimes:
            mask = labels == regime
            regime_data = features[mask]
            
            returns = regime_data['returns']
            prices = regime_data.get('close_price', None)
            
            metrics = self.risk_calc.compute_all_metrics(returns, prices)
            metrics['regime'] = int(regime)
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Calculate frequency
        total_obs = results_df['n_observations'].sum()
        results_df['frequency_pct'] = (
            results_df['n_observations'] / total_obs * 100
        )
        
        return results_df

def main():
    """Test risk metrics"""
    print("="*70)
    print("RISK ANALYTICS - Computing Metrics")
    print("="*70)
    
    # Load features
    ticker = 'SPY'
    print(f"\nLoading features for {ticker}...")
    
    features = pd.read_csv(
        f'data/processed/{ticker}_features.csv',
        index_col=0,
        parse_dates=True
    )
    
    # Load HMM regime labels
    print("Loading HMM regime results...")
    
    # We'll need to reload the labels by re-running HMM
    # For now, compute overall metrics
    
    returns = features['returns'].dropna()
    prices = features['close_price'].dropna()
    
    print(f"\nComputing risk metrics for {ticker}...")
    
    risk_calc = RiskMetrics()
    metrics = risk_calc.compute_all_metrics(returns, prices)
    
    print("\n" + "="*70)
    print("OVERALL RISK METRICS")
    print("="*70)
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key:25s}: {value:10.4f}")
        else:
            print(f"  {key:25s}: {value}")
    
    print("\n✓ Risk metrics computed successfully!")
    print("\nNext: Combine with regime labels for regime-specific analysis")

if __name__ == "__main__":
    main() 
