This project was completed as part of the **School of Quants** Python course. It focuses on constructing a dynamic portfolio of corporate bonds. The strategy is based on two key heuristics combined with systematic hyperparameter optimization and portfolio construction using the vectorbt library.

## Heuristics for Weight Adjustment

- **Volatility:**  
  The strategy calculates the standard deviation of daily returns using a rolling window. This volatility estimate is used to adjust the exposure – if volatility is high, the weight of the asset is reduced proportionally with respect to a predefined threshold.

- **Momentum:**  
  The change in asset price over a specified period (e.g., 90 or 120 days) is computed to capture price trends. If the momentum is below a preset threshold, the weight is further reduced to limit exposure during unfavorable trends.

## Hyperparameter Optimization

- **Training Period:**  
  Hyperparameter tuning is carried out on historical data from 2018 to 2022.  
- **Parameters Optimized:**  
  - `vol_window`: The window size for calculating the volatility.  
  - `mom_window`: The window size for assessing momentum.  
  - `vol_threshold`: The threshold for volatility adjustment.  
  - `mom_threshold`: The threshold for momentum adjustment.  
- **Procedure:**  
  For each combination of these hyperparameters, adjusted weights are computed and a trade simulation is executed using vectorbt. The objective is to maximize the Portfolio Sharpe ratio during the training phase, which then determines the optimal parameter set.

## Portfolio Construction on Test Data

- **Dynamic Weight Adjustment:**  
  Once the best hyperparameters are determined, these are applied to the test period (starting from 2023) to dynamically compute adjusted weights.  
- **Using vectorbt:**  
  The portfolio is constructed with vectorbt’s `Portfolio.from_orders` function. This function takes the test period price series and the dynamically adjusted weights (calculated using `adjust_weights`) as input. It simulates portfolio performance with target percentages for each asset, while taking into account factors such as trading fees and cash sharing. The function essentially executes "virtual orders" based on the provided weight signals and computes the portfolio’s value trajectory over time.

## Evaluation Metrics

The performance of the portfolio is evaluated using several key metrics:

- **Portfolio Sharpe:** 1.05  
- **Cbonds Sharpe (Benchmark):** 0.86  
- **Cumulative Return:** 186.73%  
- **Maximum Drawdown:** -6.42%  
- **Calmar Ratio:** 29.07  
- **5%-VaR (Portfolio):** -0.00607  
- **Average Intra-Portfolio Correlation (IPC):** 0.35  

These metrics provide a comprehensive view of the portfolio’s risk-adjusted returns, its downside risk, and the correlations between its constituent assets.
