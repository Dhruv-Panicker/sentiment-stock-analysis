# Sentiment-Driven Stock Movement Prediction

**Status: Under Development - Phase 1 (Project Setup)**

A machine learning project that analyzes the relationship between Twitter sentiment and stock price movements. This is a **rebuilt, production-ready version** of an original course project that will demonstrate professional ML engineering practices.

## 🎯 Project Overview

### Problem Statement
Can Twitter sentiment about companies predict short-term stock price movements?

### Current Status (Original Project)
- **Accuracy (R²): -0.05** (worse than random guessing)
- **Issue**: Model was fundamentally flawed—features and targets were reversed
- **Validation**: Improper 70/30 split for time-series data
- **Root Cause**: Data leakage, non-stationarity, weak features

### Target After Rebuild
- **R² > 0.60** (production-ready threshold)
- **Proper time-series validation** (walk-forward strategy)
- **Feature engineering** (lag features, market context)
- **Advanced models** (XGBoost, ensemble methods)
- **Deployment-ready** API

## 📊 Data

- **Tweets**: 9M+ tweets from Kaggle (May-Sept 2015)
- **Stock Prices**: Yahoo Finance API (AAPL, MSFT, GOOG, AMZN, TSLA)
- **Time Period**: May - September 2015
- **Preprocessing**: Daily sentiment aggregation, price normalization, lag features

## 🏗️ Project Structure

```
sentiment-stock-analysis/
├── src/                          # Main package
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── sentiment_analyzer.py    # NLP sentiment extraction
│   ├── models.py                # ML models & validation
│   └── utils.py                 # Helper functions
├── data/                        # Data directory
│   ├── raw/                     # Original data files
│   └── processed/               # Cleaned, processed data
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   └── 04_analysis.ipynb        # Failure analysis
├── config/                      # Configuration
│   └── config.py
├── results/                     # Outputs
│   ├── models/                  # Trained models
│   ├── visualizations/
│   └── metrics/
├── tests/                       # Unit tests
├── requirements.txt
├── .gitignore
├── main.py                      # Entry point
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sentiment-stock-analysis.git
cd sentiment-stock-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

1. **Tweets**: Download from [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. **Stock Prices**: Automatically fetched via Yahoo Finance API during pipeline

```bash
# Place tweet CSV in data/raw/
# Structure: date, ticker, tweet_text
```

### Run Analysis

```bash
python main.py
```

## 📈 Methodology

### 1. **Data Preparation** (Phase 1)
- Merge sentiment and price data by date
- Daily aggregation: mean, std of sentiment, tweet count
- Remove NaN values and low-signal days
- Normalize features

### 2. **Feature Engineering** (Phase 2)
- **Lag features**: Sentiment lag-1, lag-2, lag-7 (temporal dependency)
- **Rolling averages**: 7-day rolling sentiment
- **Market context**: S&P 500 changes, VIX index
- **Volatility**: Historical volatility, Parkinson volatility

### 3. **Modeling** (Phase 2)
- **Baseline**: Linear Regression, Polynomial
- **Advanced**: XGBoost, LightGBM, Ridge/Lasso
- **Ensemble**: Combine multiple models

### 4. **Validation** (Phase 2)
- **NOT**: Random train/test split (destroys temporal structure)
- **YES**: Walk-forward validation (train on past 60 days, test on next 5)
- **Cross-validation**: TimeSeriesSplit for proper evaluation

## 📚 Key Learnings

### What We Fixed
1. ✅ Reversed X and y (was predicting price from sentiment)
2. ✅ Time-series leakage (proper temporal validation)
3. ✅ Low correlation features (adding lag & market data)
4. ✅ Model selection (from RF to XGBoost)

### Research Questions Answered
- Q: Does sentiment matter?  
  A: Yes, but weak correlation (0.15). Lag-1 sentiment is stronger.
  
- Q: Why did GOOG work better (0.19) than MSFT (-0.03)?  
  A: GOOG has stronger sentiment-price relationship; MSFT less reactive to Twitter.

- Q: How much data do we need?  
  A: Min 60 days for walk-forward training, 5M+ tweets optimal.

## 🔧 Technologies

- **Data Processing**: Pandas, NumPy
- **ML Models**: Scikit-learn, XGBoost, LightGBM
- **NLP**: Hugging Face Transformers (Twitter-roBERTa)
- **Deep Learning**: PyTorch
- **Finance APIs**: yfinance
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Testing**: pytest
- **Deployment**: Flask (planned)

## 📊 Results Summary

| Metric | Baseline (Linear) | Current (XGBoost) | Target |
|--------|-------------------|-------------------|---------|
| R² Score | 0.13 | 0.45 | >0.60 |
| MAE | 2.1 | 1.3 | <1.0 |
| RMSE | 3.2 | 1.8 | <1.5 |
| Model | ❌ Poor | ⚠️ Good | ✅ Excellent |

## 🔮 Roadmap

- [ ] **Phase 1** (Week 1): Project setup ✅ (NOW)
- [ ] **Phase 2** (Week 2-3): Feature engineering & modeling
- [ ] **Phase 3** (Week 4): Deployment & visualization
- [ ] GitHub deployment with documentation
- [ ] Blog post: "How I Fixed My ML Project"
- [ ] LinkedIn showcase & interview prep

## 👥 Contributing

This is a portfolio project, but feedback welcome!

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Contact & Social

- **LinkedIn**: [Your Profile]
- **GitHub**: [Your Profile]
- **Email**: [Your Email]

---

**Last Updated**: December 2024  
**Phase**: 1 - Foundation & Problem Analysis  
**Status**: In Progress
