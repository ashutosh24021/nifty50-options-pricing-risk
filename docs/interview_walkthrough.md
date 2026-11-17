# Interview Walkthrough

## 90-second explanation

"The project started with the question of how well Black–Scholes can price NIFTY 50 European options when the market is moving through different volatility conditions. I implemented the Black–Scholes–Merton model, including dividend yield, and checked the analytical prices against a Cox–Ross–Rubinstein binomial tree. I also computed Delta, Gamma, Vega, Theta and Rho to understand sensitivity to spot, volatility, time and rates.

For model evaluation, I calculate historical realized volatility using a lagged rolling window, so the forecast doesn't use the same day's option premium. Then I compare BSM prices with observed option prices, and group errors by volatility regime and moneyness. That helps distinguish where a constant-volatility benchmark has larger errors, though I don't automatically assume every error is caused by the Black–Scholes formula itself.

On the risk side, I use 100,000 Monte Carlo paths for a pricing convergence check and separately simulate spot and implied-volatility shocks for a short call. I compare the 99th-percentile mark-to-market loss before and after placing an initial Delta hedge. There's also a quote-pair historical backtest for cases where I have repeated observations of the same option contract. I record the assumptions, costs and actual loss distributions, so the VaR improvement is a computed result, not a preselected number."

## Questions an interviewer may ask

**Why is Black–Scholes a benchmark and not a full description of market prices?**

It assumes a particular diffusion and constant volatility for a given contract. Actual index-option prices show changing volatility, skew and supply-demand effects, so deviations are expected. Observed option premium also carries microstructure and measurement noise.

**Did you estimate implied volatility from each premium and then compare BSM against that premium?**

No, that would be circular. The benchmark prices use **lagged rolling historical volatility**. Market-implied volatility is an independent diagnostic field calculated afterward, not the fitted input used to claim accuracy.

**What are the 100,000 simulations?**

There are separate uses of 100k samples: (1) 100k *risk-neutral terminal-price paths* to check Monte Carlo pricing against analytic BSM; (2) 100k scenarios for the short-option one-horizon VaR experiment; and (3) an optional 100k empirical bootstrap sample of observed quote residuals or paired hedge P&Ls. Bootstrap counts are resamples, not unique market observations.

**Why does Delta hedging reduce VaR but not eliminate the risk?**

Delta hedging offsets the first-order change from small spot moves. Gamma generates second-order P&L, Vega creates sensitivity to volatility shocks, and there are carry and execution costs. With a single hedge at the start, delta drift and non-linearities leave residual risk.

**Where did the 18% figure from the original resume come from?**

This new repository does not have the original input records or hedge policy, so **it does not independently verify that number**. The synthetic example produces an entirely different result and is clearly labeled. I would need the original historical option data, position book, rebalance assumptions and VaR construction to reproduce 18% in a verifiable way. Do not present this new demo as proof of the earlier figure.

**What does VaR mean here, exactly?**

It is the **99th percentile loss** on a one-horizon, single short European option with or without initial delta exposure. The report uses index points per option unit. Scenario VaR is model-dependent; historical paired-quote bootstrap VaR is a descriptive summary of the pairs in the data, not a prospective risk guarantee.

**What did the binomial tree contribute?**

It cross-validates the analytic price using a different numerical construction. Convergence at higher numbers of steps catches mistakes in discounting, risk-neutral probabilities, dividend handling and European exercise payoff logic.

**What would you improve for a production-quality study?**

Clean synchronized spot/option quote and futures data; daily risk-free/dividend curves; bid–ask and lot-size cost assumptions; train-test splits; rolling volatility surfaces; market or Heston volatility shocks; and a date-block backtest on a defined hedged portfolio rather than one short call.
