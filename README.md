# Market Regime Detection & Risk Analytics System

A comprehensive framework for detecting market regimes, performing risk analysis, and implementing regime-aware trading strategies. This project demonstrates advanced data science techniques including statistical modeling, machine learning, and quantitative finance.

## 🎯 Project Overview

This system provides a complete workflow for:
- **Regime Detection**: Multiple approaches (HMM, GMM, K-Means) to identify market regimes
- **Feature Engineering**: Comprehensive technical indicators and statistical features
- **Risk Analytics**: VaR, CVaR, Sharpe ratio, and drawdown analysis
- **Backtesting**: Regime-aware trading strategies with comprehensive performance metrics
- **Visualization**: Interactive plots and professional charts
- **Interactive Dashboard**: Gradio-based web interface (coming soon)

## 🚀 Features

### Core Capabilities

1. **Multiple Regime Detection Methods**
   - Hidden Markov Models (HMM) - Captures temporal dependencies
   - Gaussian Mixture Models (GMM) - Probabilistic regime assignment
   - K-Means Clustering - Fast regime identification

2. **Comprehensive Feature Engineering**
   - Returns and volatility (multiple timeframes)
   - Moving averages (10, 50, 200-day)
   - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
   - Momentum indicators
   - Drawdown analysis

3. **Risk Analytics**
   - Value at Risk (VaR) at multiple confidence levels
   - Conditional VaR (CVaR/Expected Shortfall)
   - Sharpe and Sortino ratios
   - Maximum drawdown and duration
   - Regime-specific risk metrics

4. **Backtesting Engine**
   - Multiple regime-aware strategies
   - Comprehensive performance metrics
   - Trade logging and statistics
   - Equity curve visualization
   - Strategy comparison framework

5. **Professional Visualizations**
   - Price charts with regime overlays
   - Regime timeline visualizations
   - Model comparison plots
   - Risk metric heatmaps
   - Backtest performance charts

## 📁 Project Structure

```
market-regimes/
├── data/
│   ├── raw/              # Raw price data (CSV files)
│   └── processed/        # Features and regime results
├── src/
│   ├── data_processing/  # Feature engineering
│   ├── models/           # Regime detection models
│   │   ├── clustering.py # K-Means & GMM
│   │   ├── hmm_model.py  # Hidden Markov Model
│   │   └── backtest.py   # Backtesting engine
│   └── risk_analytics/   # Risk metrics
├── scripts/
│   ├── explore_features.py      # Feature exploration
│   ├── visualize_regimes.py     # Regime visualizations
│   ├── run_backtest.py          # Run backtests
│   └── run_full_pipeline.py     # Complete workflow
├── figures/
│   ├── regimes/          # Regime visualization plots
│   ├── backtest/         # Backtest performance charts
│   └── risk/             # Risk analysis plots
├── app/                   # Interactive dashboard (Gradio)
├── notebooks/             # Jupyter notebooks for EDA
├── models/                # Saved model objects
└── tests/                 # Unit tests
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd market-regimes
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data** (or use your own data)
   ```bash
   python download_data.py
   ```

## 📖 Usage

### Quick Start

Run the complete pipeline:
```bash
python scripts/run_full_pipeline.py
```

This will:
1. Generate features from raw data
2. Fit all regime detection models
3. Create visualizations
4. Run risk analysis
5. Execute backtests

### Individual Steps

**1. Feature Engineering**
```bash
python src/data_processing/features.py
```

**2. Regime Detection**

K-Means & GMM:
```bash
python src/models/clustering.py
```

HMM:
```bash
python src/models/hmm_model.py
```

**3. Visualizations**
```bash
python scripts/visualize_regimes.py
```

**4. Backtesting**
```bash
python scripts/run_backtest.py
```

**5. Risk Analysis**
```bash
python src/risk_analytics/metrics.py
```

### Interactive Dashboard

Launch the Gradio interface:
```bash
python app/gradio_app.py
```

## 📊 Example Results

### Regime Detection

The system identifies 4 typical market regimes:
- **Bull Market**: High returns, low volatility
- **Moderate Growth**: Moderate returns and volatility
- **Sideways Market**: Low returns, moderate volatility
- **Bear Market**: Negative returns, high volatility

### Trading Strategies

Multiple regime-aware strategies are implemented:
- **Bull Market Only**: Trade only in bullish regimes
- **Regime Rotation**: Adjust position size based on regime
- **Momentum + Regime**: Combine momentum indicators with regime detection

### Performance Metrics

Each strategy is evaluated on:
- Total and annualized returns
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Number of trades

## 🔬 Technical Details

### Models

- **HMM**: Uses Gaussian emissions, captures regime persistence and transitions
- **GMM**: Full covariance matrix, provides probabilistic regime assignments
- **K-Means**: Fast clustering with standardized features

### Feature Engineering

- Log returns for statistical properties
- Annualized rolling volatility (multiple windows)
- Technical indicators from TA library
- Price-to-moving-average ratios
- Momentum and drawdown metrics

### Backtesting

- Walk-forward simulation
- Transaction costs (configurable commission)
- Position sizing (full/half position options)
- Comprehensive trade logging

## 📈 Key Insights

This framework demonstrates:

1. **Regime Detection Accuracy**: HMM typically provides best temporal coherence
2. **Strategy Performance**: Regime-aware strategies can outperform buy-and-hold
3. **Risk Management**: Regime-specific risk metrics help in portfolio construction
4. **Model Comparison**: Multiple models provide robustness checks

## 🎓 Educational Value

This project showcases:
- Time series analysis and feature engineering
- Statistical modeling (HMM, GMM)
- Machine learning (K-Means clustering)
- Quantitative finance concepts
- Backtesting framework design
- Data visualization best practices
- Software engineering for data science

## 📝 Configuration

Edit `config.yaml` to customize:
- Data date ranges
- Ticker symbols
- Feature parameters
- Model hyperparameters
- Risk calculation settings

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional trading strategies
- More sophisticated regime detection
- Portfolio optimization
- Real-time regime prediction
- Enhanced visualizations

## 📄 License

This project is intended for educational and research purposes.

## 🙏 Acknowledgments

- Uses `yfinance` for market data
- `hmmlearn` for HMM implementation
- `ta` library for technical indicators
- `sklearn` for clustering and preprocessing

## 📧 Contact

For questions or suggestions, please open an issue or contact the project maintainer.

---

**Note**: This is an educational project. Past performance does not guarantee future results. Use at your own risk.

