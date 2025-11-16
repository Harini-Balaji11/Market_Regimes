Market Regime Detection and Risk Analytics System
====================================================

Overview
--------
This project provides a complete framework for detecting market regimes, performing risk analysis, and generating analytical insights. 
It includes modules for feature engineering, statistical modeling, backtesting, and running an interactive dashboard.

Main Capabilities
-----------------
- Multiple regime detection approaches (Hidden Markov Models, Gaussian Mixture Models, K-Means)
- Risk analytics including VaR, CVaR, Sharpe ratio, and drawdown statistics
- Backtesting engine for regime-aware trading strategies
- Configurable Gradio-based application for interacting with results
- Support for language models to generate written insights

How to Begin
------------
1. Install all required packages:
   pip install -r requirements.txt

2. Download market data:
   python scripts/data_ingest.py

3. Open an exploratory notebook:
   jupyter notebook notebooks/01_EDA.ipynb

4. Launch the application:
   python app/gradio_app.py

Project Structure
-----------------
data/                Contains raw and processed datasets
notebooks/           Jupyter notebooks for exploration
src/                 Source code for data, models, and analytics
models/              Stored model objects
app/                 Interactive application
figures/             Plots and visual outputs
paper/               Documentation and research material
scripts/             Utility scripts
tests/               Test suite
docs/                Additional documentation

Notes
-----
This project is intended for educational and research purposes.
