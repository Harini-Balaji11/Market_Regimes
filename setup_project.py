"""
Market Regime Detection & Risk Analytics System
Project Setup Utility (Clean, Human-Readable Version)
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# Directory Structure
# ------------------------------------------------------------
PROJECT_STRUCTURE: Dict[str, List[str]] = {
    "data": ["raw", "processed", "features"],
    "notebooks": [],
    "src": ["data_processing", "models", "risk_analytics", "utils"],
    "models": ["saved_models", "checkpoints"],
    "app": ["assets", "components"],
    "figures": ["eda", "regimes", "risk", "backtest"],
    "paper": ["sections", "references"],
    "scripts": [],
    "tests": [],
    "docs": []
}

# ------------------------------------------------------------
# File Writing Utility (UTF-8 safe, but content does not use emojis)
# ------------------------------------------------------------
def write_file(path: Path, content: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"File created: {path}")
    except Exception as e:
        logging.error(f"Unable to write file {path}: {e}")


# ------------------------------------------------------------
# File Generators
# ------------------------------------------------------------
def create_requirements(base: Path):
    content = """# Core Data Science Libraries
pandas
numpy
scikit-learn
scipy

# Time Series and Regime Detection
hmmlearn
statsmodels
pomegranate

# Financial Data
yfinance
fredapi
ta
pyfolio-reloaded

# Machine Learning Models
xgboost
lightgbm

# Explainability
shap

# Visualization
matplotlib
seaborn
plotly

# Language Models
transformers
huggingface-hub
torch

# Application Framework
gradio

# Utilities
python-dotenv
tqdm
pyyaml
"""
    write_file(base / "requirements.txt", content)


def create_gitignore(base: Path):
    content = """# Python
__pycache__/
*.py[cod]
.venv/
env/

# Data
data/raw/
data/processed/
*.h5
*.pkl

# Models
models/saved_models/
models/checkpoints/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Operating System Files
.DS_Store
Thumbs.db

# Secrets
.env
config/secrets.yaml
"""
    write_file(base / ".gitignore", content)


def create_readme(base: Path):
    content = """Market Regime Detection and Risk Analytics System
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
"""
    write_file(base / "README.md", content)


def create_config(base: Path):
    content = """data:
  start_date: "2010-01-01"
  end_date: "2025-11-01"
  tickers: [SPY, QQQ, AAPL, TSLA, GLD, ^VIX]

features:
  lookback_windows: [10, 30, 60, 90]
  volatility_window: 30
  momentum_periods: [5, 10, 20]

models:
  hmm:
    n_states: 4
    n_iter: 100
    random_state: 42
  gmm:
    n_components: 4
    covariance_type: "full"
  kmeans:
    n_clusters: 4

risk:
  var_confidence: 0.95
  rolling_window: 252

hf:
  model_name: "google/flan-t5-base"
  max_length: 150
  temperature: 0.2

random_seed: 42
"""
    write_file(base / "config.yaml", content)


# ------------------------------------------------------------
# Project Creation Logic
# ------------------------------------------------------------
def create_project_structure(project_name: str):
    base_dir = Path(project_name)

    if not base_dir.exists():
        base_dir.mkdir()
        logging.info(f"Created project directory: {base_dir}")
    else:
        logging.warning(f"Directory already exists: {base_dir}")

    for parent, subdirs in PROJECT_STRUCTURE.items():
        parent_path = base_dir / parent
        parent_path.mkdir(exist_ok=True)

        for sub in subdirs:
            (parent_path / sub).mkdir(exist_ok=True)

    create_requirements(base_dir)
    create_gitignore(base_dir)
    create_readme(base_dir)
    create_config(base_dir)

    print("\nProject setup completed successfully.")
    print("Next steps:")
    print(f"1. Navigate to the project: cd {project_name}")
    print("2. Install dependencies: pip install -r requirements.txt")


# ------------------------------------------------------------
# Command Line Interface
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Initialize the Market Regime Detection project structure."
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Create the project structure"
    )

    parser.add_argument(
        "--name",
        default="market-regimes",
        help="Project name or directory name"
    )

    args = parser.parse_args()

    if args.init:
        create_project_structure(args.name)
    else:
        print("Usage: python setup_project.py --init")


if __name__ == "__main__":
    main()
