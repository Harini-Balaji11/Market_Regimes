# Market Regime Detection and Risk Analytics System

A comprehensive framework for detecting market regimes, performing risk analysis, and implementing regime-aware trading strategies. This project demonstrates advanced data science techniques including statistical modeling, machine learning, and quantitative finance.

## Project Overview

Market regimes represent distinct periods in financial markets characterized by different return and volatility patterns. For example, a bull market regime features high returns and low volatility, while a bear market regime shows negative returns and high volatility. By detecting these regimes, traders and portfolio managers can develop strategies that adapt to changing market conditions.

This system provides a complete workflow for regime detection, from feature engineering through model training, risk analysis, and strategy backtesting. The framework implements multiple detection methods, allowing you to compare approaches and choose the most appropriate one for your needs.

## Key Features

### Multiple Regime Detection Methods

The system implements three different approaches to regime detection, each with distinct advantages:

**Hidden Markov Models (HMM)**: This method captures temporal dependencies between regimes, meaning it understands how regimes evolve over time. The model provides transition probabilities showing how likely it is to move from one regime to another, as well as regime persistence metrics. This makes HMM particularly useful when you need to understand regime dynamics and predict regime changes.

**Gaussian Mixture Models (GMM)**: GMM provides probabilistic regime assignment, meaning it can express uncertainty about which regime the market is in. Rather than hard classifications, GMM gives you probabilities for each regime at any given time. This is useful when regime boundaries are not clearly defined or when you want to account for uncertainty in your analysis.

**K-Means Clustering**: This is a fast and efficient method for regime identification based on feature similarity. It groups similar market periods together without considering temporal order. K-Means is excellent for initial analysis and provides a baseline for comparison with more sophisticated methods.

### Comprehensive Feature Engineering

The feature engineering module creates 26 distinct features from raw price data. These features capture different aspects of market behavior that may indicate regime changes:

- Return measures: Log returns and simple returns at daily frequencies
- Volatility measures: Annualized rolling volatility across multiple timeframes (10, 30, 60, 90 days)
- Moving averages: 10-day, 50-day, and 200-day moving averages, plus price-to-moving-average ratios
- Technical indicators: RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), Bollinger Bands, and Average True Range (ATR)
- Momentum indicators: Rate of change across various lookback periods (5, 10, 20 days)
- Drawdown analysis: Measures capturing how far prices have fallen from their peaks

All features are carefully designed to be stationary where appropriate and to capture meaningful patterns in market data.

### Risk Analytics

The system computes comprehensive risk metrics both overall and per-regime. This allows you to understand how risk characteristics differ across market regimes:

- Value at Risk (VaR): Maximum expected loss at 95% and 99% confidence levels
- Conditional VaR (CVaR): Expected loss given that loss exceeds VaR, providing a more conservative risk measure
- Risk-adjusted returns: Sharpe ratio and Sortino ratio, with the Sortino ratio focusing only on downside volatility
- Drawdown metrics: Maximum drawdown and drawdown duration to understand peak-to-trough declines
- Regime-specific risk profiles: Detailed risk characteristics for each identified regime

These metrics help you understand not just overall portfolio risk, but how risk varies across different market conditions.

### Backtesting Engine

The backtesting framework implements multiple regime-aware trading strategies:

**Bull Market Only Strategy**: This strategy only trades when the market is identified as being in a bullish regime. It exits positions when the regime changes to avoid bear markets.

**Regime Rotation Strategy**: This strategy adjusts position sizing based on the current regime. It might take full positions in bull markets, reduce positions in moderate markets, and exit completely in bear markets.

**Momentum Plus Regime Strategy**: This combines momentum indicators with regime detection, only taking positions when both conditions are favorable - the market is in a bullish regime and showing positive momentum.

Each strategy is evaluated on comprehensive performance metrics including total return, annualized return, Sharpe ratio, maximum drawdown, number of trades, and win rate. The backtesting engine accounts for transaction costs and provides detailed trade logs for analysis.

### Professional Visualizations

The system generates over 60 professional visualizations suitable for presentations and analysis:

- Price charts with regime overlays showing how identified regimes map to actual price movements
- Regime timeline visualizations providing a compact view of regime changes over time
- Model comparison plots allowing side-by-side evaluation of different detection methods
- Risk-return scatter plots by regime showing the risk-return characteristics of each regime
- Risk metric comparisons across regimes using bar charts and other visualizations
- Backtest performance charts including equity curves, drawdown charts, and trade markers

All visualizations are saved as high-resolution PNG files suitable for reports and presentations.

## Project Structure

```
market-regimes/
├── data/
│   ├── raw/              Raw price data in CSV format
│   └── processed/        Processed features and regime analysis results
├── src/
│   ├── data_processing/  Feature engineering modules
│   ├── models/           Regime detection models and backtesting
│   │   ├── clustering.py K-Means and GMM implementations
│   │   ├── hmm_model.py  Hidden Markov Model implementation
│   │   └── backtest.py   Backtesting engine and trading strategies
│   └── risk_analytics/   Risk metric calculations
├── scripts/
│   ├── explore_features.py      Feature exploration and statistics
│   ├── visualize_regimes.py     Generate regime visualizations
│   ├── analyze_regime_risk.py   Risk analysis per regime
│   ├── compare_models.py        Compare different detection methods
│   ├── run_backtest.py          Execute backtests on all tickers
│   └── run_full_pipeline.py     Run complete end-to-end workflow
├── figures/
│   ├── regimes/          Regime visualization plots (40+ files)
│   ├── backtest/         Backtest performance charts
│   └── risk/             Risk analysis visualizations
├── app/                   Interactive dashboard (under development)
├── notebooks/             Jupyter notebooks for exploratory analysis
├── models/                Saved trained model objects
└── tests/                 Unit tests
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**
   
   If you have the repository URL:
   ```bash
   git clone <repository-url>
   cd market-regimes
   ```
   
   Otherwise, ensure you have the project files in a directory called `market-regimes`.

2. **Install required packages**
   
   All required dependencies are listed in `requirements.txt`. Install them with:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install packages including pandas, numpy, scikit-learn, hmmlearn, ta, matplotlib, and others.

3. **Download market data**
   
   The system needs historical price data to work. You can download data using:
   ```bash
   python download_data.py
   ```
   
   This will download data for SPY, QQQ, AAPL, and TSLA from 2010 to present. You can also use your own data files in the `data/raw/` directory.

## Usage

### Running the Complete Pipeline

The easiest way to run everything is to use the master pipeline script:

```bash
python scripts/run_full_pipeline.py
```

This script will prompt you to confirm, then execute all steps automatically:
1. Feature engineering from raw price data
2. Fitting all regime detection models (K-Means, GMM, HMM)
3. Creating regime visualizations
4. Running risk analysis
5. Executing backtests for all tickers

The entire process typically takes 5-10 minutes depending on your system. All results are saved to the appropriate directories.

### Running Individual Components

If you prefer to run steps individually or need to re-run specific components:

**Feature Engineering**

Generate features from raw price data:
```bash
python src/data_processing/features.py
```

This creates feature CSV files in `data/processed/` with 26 features for each ticker.

**Regime Detection - Clustering Models**

Run K-Means and Gaussian Mixture Model detection:
```bash
python src/models/clustering.py
```

This fits both models and saves regime statistics to CSV files.

**Regime Detection - Hidden Markov Model**

Run HMM-based regime detection:
```bash
python src/models/hmm_model.py
```

This fits the HMM model and generates transition probability matrices showing how regimes evolve.

**Create Visualizations**

Generate all regime visualization plots:
```bash
python scripts/visualize_regimes.py
```

This creates price charts with regime overlays, timeline visualizations, statistics plots, and model comparison charts for all tickers. Results are saved to `figures/regimes/`.

**Risk Analysis**

Perform regime-aware risk analysis:
```bash
python scripts/analyze_regime_risk.py
```

This computes risk metrics (VaR, CVaR, Sharpe ratio, drawdown) for each regime and creates risk-return scatter plots and comparison charts.

**Model Comparison**

Compare all three detection methods:
```bash
python scripts/compare_models.py
```

This provides side-by-side comparison of K-Means, GMM, and HMM, showing agreement metrics and recommendations.

**Backtesting**

Run regime-aware trading strategies:
```bash
python scripts/run_backtest.py
```

This tests all strategies on all tickers and generates performance metrics and equity curve charts.

## Example Results

### Regime Detection Results

The system typically identifies four distinct market regimes:

- **Bull Market**: Characterized by positive returns and low to moderate volatility. This is the most favorable regime for long positions.

- **Moderate Growth**: Features positive returns but with higher volatility than bull markets. Still favorable but requires more risk management.

- **Sideways Market**: Shows low returns (slightly positive or negative) with moderate volatility. Markets are essentially flat, making trading more challenging.

- **Bear Market**: Characterized by negative returns and high volatility. This regime poses the greatest risk and strategies typically exit or short positions.

The exact characteristics depend on the ticker and time period analyzed. The system automatically identifies these patterns from the data.

### Trading Strategy Performance

Example results from backtesting show how different strategies perform:

The Bull Market Only strategy typically achieves good risk-adjusted returns by avoiding bear markets. For example, on SPY from 2010-2024, this strategy achieved approximately 120% total return with a 60% win rate.

The Regime Rotation strategy adjusts exposure based on regime, potentially improving returns during transitions but requiring careful execution.

The Momentum Plus Regime strategy combines multiple signals, generating more trades but with potentially lower win rates.

Results vary significantly by ticker and time period. Individual stocks like TSLA show higher volatility and more dramatic regime changes than broad market ETFs like SPY.

### Performance Metrics Explained

Each strategy is evaluated on several key metrics:

- **Total Return**: Overall percentage gain or loss over the entire period
- **Annual Return**: Average annualized return, useful for comparing strategies across different time periods
- **Sharpe Ratio**: Risk-adjusted return measure. Values above 1.0 are considered good, above 2.0 are excellent
- **Maximum Drawdown**: Largest peak-to-trough decline, expressed as a negative percentage
- **Win Rate**: Percentage of trades that were profitable
- **Number of Trades**: Total trading frequency, important for understanding strategy activity

These metrics together provide a comprehensive picture of strategy performance, accounting for both returns and risk.

## Technical Details

### Model Implementation Details

**Hidden Markov Model**: The HMM implementation uses Gaussian emissions, meaning each regime is modeled as a multivariate Gaussian distribution over the feature space. The model learns both the emission parameters (mean and covariance for each regime) and the transition probabilities between regimes. Training uses the Baum-Welch algorithm (expectation-maximization) to find optimal parameters.

**Gaussian Mixture Model**: GMM uses a full covariance matrix, allowing it to capture correlations between features within each regime. The model provides both hard labels (most likely regime) and soft probabilities (probability distribution over all regimes). This probabilistic output is useful for understanding uncertainty.

**K-Means Clustering**: K-Means uses standardized features (zero mean, unit variance) to ensure all features contribute equally to distance calculations. The algorithm runs multiple initializations to avoid local optima and uses the silhouette score and Davies-Bouldin index to evaluate clustering quality.

### Feature Engineering Approach

The feature engineering follows several principles:

- **Log returns**: Used instead of simple returns because they have better statistical properties, are approximately normally distributed, and are time-additive.

- **Multiple timeframes**: Volatility and momentum features are computed at multiple windows to capture both short-term and long-term patterns.

- **Technical indicators**: Standard indicators from the TA library are included to capture momentum, trend, and volatility signals that traders commonly use.

- **Price-to-MA ratios**: These provide relative strength measures showing how current prices compare to moving averages.

All features are carefully validated to ensure they contain useful information and are not redundant.

### Backtesting Methodology

The backtesting engine implements a walk-forward simulation:

- **Transaction costs**: A configurable commission rate (default 0.1%) is applied to all trades to simulate realistic trading costs.

- **Position sizing**: Strategies can specify position sizes as a fraction of available capital, allowing for full or partial positions.

- **Trade execution**: Trades are executed at the close price on the signal date, simulating realistic execution.

- **Equity tracking**: The system tracks equity curve (portfolio value over time) including both realized and unrealized gains.

- **Trade logging**: Every trade is logged with entry date, exit date, entry price, exit price, profit/loss, holding period, and regime at entry.

The backtesting framework ensures that signals from one day are executed at the next available price, avoiding look-ahead bias.

## Configuration

The system uses a configuration file (`config.yaml`) to centralize settings:

- **Data settings**: Date ranges and ticker symbols to analyze
- **Feature parameters**: Lookback windows, volatility windows, momentum periods
- **Model hyperparameters**: Number of regimes, random seeds, iteration limits
- **Risk settings**: VaR confidence levels, rolling window sizes

You can modify this file to customize the analysis without changing code. See `config.yaml` for current settings.

## Key Insights and Findings

This framework demonstrates several important concepts:

1. **Regime Detection Validity**: Different models can identify similar regimes but with different temporal structures. HMM typically provides the most coherent temporal sequence, while K-Means can be more sensitive to feature scaling.

2. **Strategy Effectiveness**: Regime-aware strategies can outperform simple buy-and-hold by avoiding unfavorable market conditions. However, transaction costs and regime identification accuracy significantly impact results.

3. **Risk Heterogeneity**: Risk characteristics vary dramatically across regimes. Bear markets show much higher VaR and drawdowns than bull markets, highlighting the importance of regime-aware risk management.

4. **Model Robustness**: Comparing multiple models provides validation that detected regimes are not artifacts of a single method. High agreement between models increases confidence in the results.

These insights are valuable for both understanding market behavior and developing trading strategies.

## Educational Value

This project demonstrates several important skills and concepts:

- **Time Series Analysis**: Working with financial time series data, handling missing values, dealing with non-stationarity

- **Feature Engineering**: Creating meaningful features from raw data, understanding which features matter for different problems

- **Statistical Modeling**: Implementing and tuning statistical models like HMM and GMM

- **Machine Learning**: Applying clustering algorithms to real-world data

- **Quantitative Finance**: Understanding market regimes, risk metrics, and trading strategy development

- **Backtesting**: Building realistic backtesting frameworks that avoid common pitfalls like look-ahead bias

- **Data Visualization**: Creating clear, informative visualizations for analysis and presentation

- **Software Engineering**: Organizing code into modules, writing reusable functions, handling errors gracefully

These skills are applicable across many data science and finance domains.

## Acknowledgments

This project uses several excellent open-source libraries:

- **yfinance**: For downloading market data from Yahoo Finance
- **hmmlearn**: For Hidden Markov Model implementation
- **ta**: For technical analysis indicators
- **scikit-learn**: For clustering algorithms, preprocessing, and utility functions
- **pandas and numpy**: For data manipulation and numerical computations
- **matplotlib and seaborn**: For creating visualizations

We are grateful to the maintainers and contributors of these projects.

**Important Note**: This is an educational project demonstrating quantitative finance and machine learning concepts. The results shown are from historical backtesting and should not be interpreted as investment advice. Always conduct your own research and consider professional financial advice before making investment decisions. Past performance does not guarantee future results.
