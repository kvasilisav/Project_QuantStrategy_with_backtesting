import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def sharpe_ratio(returns, rf=0):
    return np.sqrt(252) * (returns.mean() - rf) / returns.std()

def calmar_ratio(returns, printing = True):
    cum_return = (1 + returns).cumprod()
    drawdown = cum_return / cum_return.cummax() - 1
    max_drawdown = drawdown.min()
    total_return = cum_return.iloc[-1] - 1
    calmar = (cum_return.iloc[-1] - 1) / abs(max_drawdown)
    if printing:
        print(f"Накопленная доходность: {total_return*100:0.2f}%")
        print(f"Максимальная просадка: {max_drawdown*100:0.2f}%")
        print(f"Коэффициент Калмара: {calmar:0.2f}")
    return total_return, max_drawdown, calmar

def calc_ipc(prices, printing = True):
    ipc = prices.pct_change().corr().mean().mean()
    if printing:
        print(f"Средняя внутрипортфельная корреляция (IPC): {ipc:0.2f}")
    return ipc

def calc_var_with_pca(pf_returns, printing = True):
    pca = PCA(n_components=1)
    pca_fit = pca.fit_transform(pf_returns.values.reshape(-1, 1))
    var_5 = np.percentile(pf_returns, 5)
    if printing:
        print("5%-ный VaR портфеля:", var_5)
    return var_5

def bootstrap_sharpe(returns, n_bootstraps=1000):
    bootstrapped = []
    n = len(returns)
    for _ in range(n_bootstraps):
        sample = np.random.choice(returns, size=n, replace=True)
        bootstrapped.append(sharpe_ratio(pd.Series(sample)))
    return np.array(bootstrapped)