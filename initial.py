# %%
import yfinance as yf
import pandas as pd
import numpy as np
import datetime


# %%
today = datetime.date.today().strftime("%Y-%m-%d")
data = yf.download("SPY", start="2001-01-01", end=today)
open('SPY.csv', 'w').close()
data.to_csv('SPY.csv')

data.columns = ['c','o','l','h','v']
#print(data['c'].head())
log_returns_1 = np.log(data['c'] / data['c'].shift(1))
#print(log_returns_1.head())
log_returns_1.to_csv('SPY_log_returns_1.csv')

# %%
data['log_return_1'] = np.log(data['c'] / data['c'].shift(1))
data['log_return_5'] = np.log(data['c'] / data['c'].shift(5))
data['log_return_21'] = np.log(data['c'] / data['c'].shift(21))
data['vol_21'] = data['log_return_1'].rolling(21).std()
data['target'] = (data['log_return_1'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)
data.columns

# %%
from ta.momentum import RSIIndicator
data['rsi'] = RSIIndicator(data['c'], window=14).rsi()
data['ma_ratio'] = data['c'] / data['c'].rolling(50).mean()
data.dropna(inplace=True)
data.columns
print(len(data))


# %%
features = ['log_return_1', 'log_return_5', 'log_return_21', 'vol_21', 'rsi', 'ma_ratio']
x = data[features]
y = data['target']

split = int(len(data) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

print(x_train.size, y_train.size)

# %%
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(x_train, y_train)

# %%
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%
import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(model)
plt.show()

# %%
vix_data = yf.download("^VIX", start="2001-01-01", end=today)
vix_data.to_csv('VIX.csv')

# %%
vix_pct_change = vix_data['Close'].pct_change()
vix_data['vix_pct_change'] = vix_pct_change
vix_data.dropna(inplace=True)

# %%
vix_data['vix_pct_change'].head()

# %%
data['v_change'] = data['c'].pct_change()
data['dollar_volume'] = data['c'] * data['v']
data.dropna(inplace=True)


# %%
data['overnight_return'] = np.log(data['o'] / data['c'].shift(1))
data['day_of_week'] = pd.to_datetime(data.index).dayofweek
data.dropna(inplace=True)


# %%
data['vix_pct_change'] = vix_data['vix_pct_change']

# %%
print(data.columns)

# %%
features = ['log_return_1', 'log_return_5', 'log_return_21', 'vol_21', 'rsi', 'ma_ratio', 'v_change', 'dollar_volume', 'overnight_return', 'day_of_week', 'vix_pct_change']
data.dropna(inplace=True)

# %%
x = data[features]
y = data['target']

split = int(len(data) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(x_train, y_train)


# %%
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
y_prob = model.predict_proba(x_test)[:, 1]

print(y_prob[:5])

# %%
signal = (y_prob < 0.6).astype(int)

# %%
test_returns = data['log_return_1'].iloc[split:]
strategy_returns =  signal * test_returns

print(strategy_returns.head())


# %%
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f'Sharpe Ratio: {sharpe}')

# %%
plt.figure(figsize=(12,5))
plt.plot((1 + strategy_returns).cumprod(), label='Strategy')
plt.plot((1 + test_returns).cumprod(), label='Buy & Hold')
plt.legend()
plt.show()

# %%
cum_return = (1+strategy_returns).cumprod()
cum_max = cum_return.cummax()
drawdown = (cum_return - cum_max)/cum_max

max_drawdown = drawdown.min()
print(f'Max Drawdown: {max_drawdown:.2%}')

# %%
trades = (pd.Series(signal).diff().abs().sum())

print(f'Number of trades: {trades}')

# %%
signal = pd.Series(signal, index=x_test.index)
transaction_costs = pd.Series(signal.diff().abs()) * 0.001
strategy_returns_net = strategy_returns - transaction_costs

# %%
sharpe_net = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
print(f'Sharpe Ratio (net of costs): {sharpe_net}')

max_drawdown_net = (strategy_returns_net.cumsum() - strategy_returns_net.cumsum().cummax()).min()
print(f'Max Drawdown (net of costs): {max_drawdown_net:.2%}')

