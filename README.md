# Global Market Intelligence AI Agent

An AI-powered gold market research dashboard designed to explore market regimes, analyze historical price behavior, and review forecasting and backtesting results.

## Overview

GlobalMarketAI is a Python-based research project focused on gold market analysis. It combines historical market data, technical indicators, machine learning experiments, and strategy evaluation tools in one project.

## Features

- **Market Dashboard:** Interactive visualization of gold market data.
- **Market Regime Analysis:** Examine different market conditions and regime transitions.
- **Gold Price Forecasting:** Review model-generated predictions and analysis.
- **Technical Analysis:** Explore support and resistance levels.
- **Backtesting:** Evaluate historical strategy performance and trading assumptions.
- **Model Validation:** Analyze prediction quality, probability calibration, and walk-forward results.
- **Research Tools:** Python scripts for data audits, feature engineering, and model comparisons.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Yahoo Finance data through yfinance

## Project Structure

```text
GlobalMarketAI/
├── market_dashboard.py
├── app.py
├── market_data.py
├── market_regime.py
├── features.py
├── train_model.py
├── predict_latest_gold.py
├── backtest.py
├── requirements.txt
├── gold_data.csv
├── gold_5m_data.csv
├── gold_market_regimes.csv
├── latest_gold_prediction.csv
└── Other analysis, validation, and backtesting scripts
```

The exact files available may change as the project develops.

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd GlobalMarketAI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run market_dashboard.py
```

Open the local URL displayed in your terminal.

## Research and Limitations

This project is intended for educational purposes, market research, and experimental model evaluation. Historical performance and predictive metrics do not guarantee future results.

Models and strategies require further independent validation. This project should not be treated as a source of guaranteed trading signals or financial advice.

## Future Improvements

- Automated market-data refresh
- Improved model validation across market conditions
- Additional forecasting horizons
- More robust risk and transaction-cost analysis
- Automated dashboard updates

## License

No license has been selected yet. All rights remain with the respective copyright holders unless a license is added.
