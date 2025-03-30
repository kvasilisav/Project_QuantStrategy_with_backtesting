import pandas as pd
import numpy as np 
import vectorbt as vbt
from itertools import product
from metrics import sharpe_ratio


def get_prices(quotes_dirty: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the 'quotes_dirty' DataFrame into a clean price DataFrame.

    This function pivots the provided DataFrame to have dates as rows and ISINs as columns,
    using the 'dirty_price' column as values. It applies forward and backward filling to ensure
    no missing values, replaces non-positive prices with a small positive constant (1e-3), and
    asserts that all prices are strictly positive.

    Parameters:
    - quotes_dirty (pd.DataFrame): DataFrame containing at least 'Trade date', 'ISIN', and 'dirty_price'.

    Returns:
    - pd.DataFrame: A cleaned and pivoted DataFrame with strict positive prices.
    """
    prices = quotes_dirty.pivot(index="Trade date", columns="ISIN", values="dirty_price").sort_index().ffill().bfill()
    min_positive = 1e-3
    prices = prices.mask(prices <= 0, min_positive)
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    assert (prices > 0).all().all(), "Некоторые цены не положительны"
    return prices



def get_weights(structure: pd.DataFrame, quotes_dirty: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the weight matrix based on the bond index structure.

    The function first aggregates the 'Issue amount, USDmn' by the index structure date.
    Then, for each record in the structure DataFrame, it computes the weight of each bond
    by dividing its issue amount by the total issue amount on the same structure date.
    The resulting weight DataFrame is pivoted, aligned with the quotes dates and prices, and 
    missing data is filled with zeros.

    Parameters:
    - structure (pd.DataFrame): DataFrame containing the bond index structure, including 'Index Structure Date',
      'ISIN', and 'Issue amount, USDmn'.
    - quotes_dirty (pd.DataFrame): DataFrame with bond quotes used to extract unique trade dates.
    - prices (pd.DataFrame): DataFrame of cleaned prices aligned by trade date and ISIN.

    Returns:
    - pd.DataFrame: A DataFrame of bond weights, aligned with the index of 'prices' and columns representing ISINs.
    """
    amount = structure.groupby("Index Structure Date")["Issue amount, USDmn"].sum()
    structure["weights"] = structure.apply(
        lambda x: x["Issue amount, USDmn"] / amount.loc[x["Index Structure Date"]],
        axis=1,
    )
    w = structure[["Index Structure Date", "ISIN", "weights"]].set_index(["Index Structure Date", "ISIN"])
    w = w[~w.index.duplicated()].unstack().fillna(0)
    w = w.reindex(quotes_dirty["Trade date"].unique(), method="ffill")
    w = w["weights"]
    w_aligned = w.reindex(index=prices.index, columns=prices.columns, fill_value=0).fillna(method="ffill")
    return w_aligned



def adjust_weights(weights: pd.DataFrame, prices: pd.DataFrame, 
                   vol_window=30, mom_window=90, vol_threshold=0.05, mom_threshold=0.0) -> pd.DataFrame:
    """
    Adjust weights dynamically using volatility and momentum factors.

    The function applies a two-step adjustment on the given weights. First, it calculates daily returns
    and their rolling volatility over a specified window. A volatility adjustment factor is computed as
    the ratio between the given volatility threshold and the rolling volatility, clipped to an upper limit of 1,
    and then shifted by one day. Secondly, a momentum factor is calculated over the momentum window and applied,
    where if the momentum is below a given threshold, a constant factor of 0.8 is used instead.
    Finally, weights are re-normalized across each day.

    Parameters:
    - weights (pd.DataFrame): DataFrame of initial weights.
    - prices (pd.DataFrame): DataFrame of asset prices to compute returns.
    - vol_window (int): The rolling window size for volatility calculation (default is 30).
    - mom_window (int): The window size to calculate momentum (default is 90).
    - vol_threshold (float): Threshold for volatility-based adjustment (default is 0.05).
    - mom_threshold (float): Minimum momentum threshold (default is 0.0).

    Returns:
    - pd.DataFrame: The adjusted weights DataFrame after applying volatility and momentum factors.
    """
    daily_returns = prices.pct_change()
    vol = daily_returns.rolling(window=vol_window).std()
    momentum = (prices / prices.shift(mom_window)) - 1
    
    adjusted_weights = weights.copy()
    vol_factor = vol_threshold / vol.replace(0, np.nan)
    vol_factor = vol_factor.clip(upper=1) 
    adjusted_weights = adjusted_weights.multiply(vol_factor.shift(1))  

    mom_factor = 1 + momentum.shift(1)  
    mom_factor = mom_factor.where(momentum.shift(1) >= mom_threshold, 0.8)  
    adjusted_weights = adjusted_weights.multiply(mom_factor)
    adjusted_weights = adjusted_weights.div(adjusted_weights.sum(axis=1), axis=0).fillna(0)
    
    return adjusted_weights


def train_model(train_start, train_end, test_start, prices, structure, quotes_dirty, commission, 
                vol_window_candidates,mom_window_candidates, vol_threshold_candidates, mom_threshold_candidates, 
                printing = True):
  """
    Train and select the best hyperparameters based on the training period and apply them on the test set.

    The function splits the price data into training and testing sets based on the specified dates.
    For the training set, it computes the weight matrix and iterates over combinations of volatility and 
    momentum window sizes along with their corresponding thresholds to adjust the weights. For each combination,
    a portfolio is simulated using vectorbt, and the Sharpe ratio is calculated. The best combination is selected
    based on the highest Sharpe ratio. Finally, the selected hyperparameters are used together with the test data,
    and the adjusted weights for the test period are computed.

    Parameters:
    - train_start (str): Start date for the training period.
    - train_end (str): End date for the training period.
    - test_start (str): Start date for the test period.
    - prices (pd.DataFrame): DataFrame of asset prices.
    - structure (pd.DataFrame): DataFrame of bond index structure.
    - quotes_dirty (pd.DataFrame): DataFrame with dirty prices and additional bond quote data.
    - commission (float): Transaction fees to be used in portfolio simulation.
    - vol_window_candidates (iterable): Candidate values for the volatility window.
    - mom_window_candidates (iterable): Candidate values for the momentum window.
    - vol_threshold_candidates (iterable): Candidate values for the volatility threshold.
    - mom_threshold_candidates (iterable): Candidate values for the momentum threshold.
    - printing (bool): If True, prints the best hyperparameters from the training period (default True).

    Returns:
    - tuple: A tuple containing:
       - best_params (pd.Series): The hyperparameters with the highest training Sharpe ratio.
       - prices_test (pd.DataFrame): Subset of the price DataFrame for the test period.
       - adjusted_weights_test (pd.DataFrame): Adjusted weights computed on the test period.
    """
  prices_train = prices.loc[train_start:train_end]
  w_aligned_train = get_weights(structure, quotes_dirty, prices_train)
  results = []
  for vol_win, mom_win, vol_thr, mom_thr in product(vol_window_candidates,
                                                  mom_window_candidates,
                                                  vol_threshold_candidates,
                                                  mom_threshold_candidates):
      adjusted_weights_train = adjust_weights(
          w_aligned_train, prices_train,
          vol_window=vol_win, mom_window=mom_win,
          vol_threshold=vol_thr, mom_threshold=mom_thr
      )

      pf_train = vbt.Portfolio.from_orders(
          close=prices_train,
          size=adjusted_weights_train,
          size_type="targetpercent",
          direction="longonly",
          freq='1D',
          fees=commission,
          cash_sharing=True
      )
      
      pf_value_train = pf_train.value()
      pf_returns_train = pf_value_train.pct_change().fillna(0)
      pf_sharpe_train = sharpe_ratio(pf_returns_train)
      
      results.append({
          "vol_window": vol_win,
          "mom_window": mom_win,
          "vol_threshold": vol_thr,
          "mom_threshold": mom_thr,
          "train_sharpe": pf_sharpe_train
      })
  results_df = pd.DataFrame(results)
  best_params = results_df.sort_values(by="train_sharpe", ascending=False).iloc[0]
  if printing:
    print("Лучшие гиперпараметры по обучающему периоду (2018-2022):")
    print(best_params)

  prices_test = get_prices(quotes_dirty).loc[test_start:]
  w_aligned_test = get_weights(structure, quotes_dirty, prices_test)
  adjusted_weights_test = adjust_weights(
      w_aligned_test, prices_test,
      vol_window=int(best_params.loc['vol_window']), mom_window= int(best_params.loc['mom_window']),
      vol_threshold=best_params.loc['vol_threshold'], mom_threshold=best_params.loc['mom_threshold']
  )
  return best_params, prices_test, adjusted_weights_test

