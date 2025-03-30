import pandas as pd
import numpy as np 
import vectorbt as vbt
from itertools import product
from metrics import sharpe_ratio


def get_prices(quotes_dirty):
    prices = quotes_dirty.pivot(index="Trade date", columns="ISIN", values="dirty_price").sort_index().ffill().bfill()
    min_positive = 1e-3
    prices = prices.mask(prices <= 0, min_positive)
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    assert (prices > 0).all().all(), "Некоторые цены не положительны"
    return prices



def get_weights(structure, quotes_dirty, prices):
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



def adjust_weights(weights, prices, vol_window=30, mom_window=90, vol_threshold=0.05, mom_threshold=0.0):
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

