import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def sharpe_ratio(returns, rf=0):
    """
    Calculate the annualized Sharpe ratio for a series of returns.

    Parameters:
    - returns (pd.Series or np.array): Series or array of asset returns.
    - rf (float): Risk-free rate (default is 0).

    Returns:
    - float: The annualized Sharpe ratio.
    """
    return np.sqrt(252) * (returns.mean() - rf) / returns.std()

def calmar_ratio(returns, printing = True):    
    """
    Calculate the cumulative return, maximum drawdown, and Calmar ratio of a return series.
    The function computes:
    - Cumulative return using cumulative product.
    - Maximum drawdown by comparing the current cumulative return with the running maximum.
    - Calmar ratio as the ratio between the total return and the absolute maximum drawdown.

    Parameters:
    - returns (pd.Series): Series of asset returns.
    - printing (bool): If True, prints the cumulative return, maximum drawdown, and Calmar ratio.

    Returns:
    - tuple: (total_return, max_drawdown, calmar)
    """
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
    """
    Calculate the average intra-portfolio correlation (IPC).

    The function computes the percentage change of prices, then the correlation matrix of these changes,
    and finally calculates the mean correlation value by averaging over the entire matrix.

    Parameters:
    - prices (pd.DataFrame): DataFrame with asset prices.
    - printing (bool): If True, prints the computed IPC.

    Returns:
    - float: The average intra-portfolio correlation.
    """
    ipc = prices.pct_change().corr().mean().mean()
    if printing:
        print(f"Средняя внутрипортфельная корреляция (IPC): {ipc:0.2f}")
    return ipc

def calc_var_with_pca(pf_returns, printing = True):
    """
    Calculate the Value at Risk (VaR) of a portfolio using PCA.

    This function performs PCA on the portfolio returns (reshaped into a single feature) and calculates the 
    5th percentile (VaR at 5%) of the returns distribution.

    Parameters:
    - pf_returns (pd.Series): Series of portfolio returns.
    - printing (bool): If True, prints the 5% VaR of the portfolio.

    Returns:
    - float: The 5% Value at Risk (VaR) of the portfolio.
    """
    pca = PCA(n_components=1)
    pca_fit = pca.fit_transform(pf_returns.values.reshape(-1, 1))
    var_5 = np.percentile(pf_returns, 5)
    if printing:
        print("5%-ный VaR портфеля:", var_5)
    return var_5

def bootstrap_sharpe(returns, n_bootstraps=1000):
    """
    Bootstrap the Sharpe ratio for a series of returns.

    Given a return series, this function resamples the returns with replacement multiple times to generate 
    a distribution of Sharpe ratios. It is useful for assessing the variability and confidence intervals of the Sharpe ratio.

    Parameters:
    - returns (iterable): Array-like series of asset returns.
    - n_bootstraps (int): Number of bootstrap samples (default is 1000).

    Returns:
    - np.array: An array containing the bootstrapped Sharpe ratios.
    """
    bootstrapped = []
    n = len(returns)
    for _ in range(n_bootstraps):
        sample = np.random.choice(returns, size=n, replace=True)
        bootstrapped.append(sharpe_ratio(pd.Series(sample)))
    return np.array(bootstrapped)