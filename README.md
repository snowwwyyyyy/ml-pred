# ML Price Prediction for Forex Trading

Machine learning models (LSTM, Random Forest, XGBoost) for predicting forex price movements using technical indicators and historical price data.

## Model Overview

- **Models**: LSTM (Deep Learning), Random Forest, XGBoost
- **Features**: Technical indicators (RSI, MACD, Bollinger Bands, ATR, Moving Averages)
- **Target**: Next-day price movement (classification) or price value (regression)
- **Data**: USD/EUR forex pairs (2018-2024)
- **Goal**: Predict directional movement for trading decisions

## Pipeline

1. **Data Collection**: Historical OHLCV data via yfinance
2. **Feature Engineering**: Calculate 15+ technical indicators
3. **Model Training**: Train LSTM, Random Forest, and XGBoost
4. **Hyperparameter Tuning**: Grid search for optimal parameters
5. **Backtesting**: Simulate trading with model predictions
6. **Evaluation**: Accuracy, precision, recall, Sharpe ratio

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LSTM | 54% | 0.53 | 0.58 | 0.55 |
| Random Forest | 52% | 0.51 | 0.54 | 0.52 |
| XGBoost | 53% | 0.52 | 0.55 | 0.53 |

**Key Findings:**
- All models slightly better than random (50%)
- LSTM performs best on trending markets
- Feature importance: ATR, RSI, and volume-based indicators rank highest
- High noise-to-signal ratio in forex data limits prediction accuracy

## Tech Stack

- **Python 3.11**
- **TensorFlow/Keras** - LSTM implementation
- **scikit-learn** - Random Forest, preprocessing
- **XGBoost** - Gradient boosting
- **pandas, numpy** - Data manipulation
- **TA-Lib / pandas-ta** - Technical indicators

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/train_lstm.py
python src/train_rf.py

# Run backtest
python src/backtest.py
```

## Key Learnings

- **Feature importance**: ATR and RSI provide strongest signals
- **LSTM advantages**: Captures temporal dependencies in sequential data
- **Overfitting risk**: Regularization and dropout essential for generalization
- **Market efficiency**: Forex price prediction remains challenging due to high noise

## Future Enhancements

- [ ] Sentiment analysis from news/Twitter
- [ ] Multi-timeframe feature aggregation
- [ ] Ensemble methods combining all 3 models
- [ ] Real-time prediction API
- [ ] Reinforcement learning for dynamic position sizing

## Author

**Snow**  
VNIT Nagpur  
snowj3327@gmail.com

---

*Developed as part of quantitative finance learning. Demonstrates ML workflow for financial time series forecasting.*
