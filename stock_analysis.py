# build frontier of four stocks
# a tutorial from 'https://towardsdatascience.com/efficient-frontier-optimize-portfolio-with-scipy-57456428323e'
########################################################################
# compute the risk return
# K * (expected return – expected risk free rate)
# K is the square root of frequency of sampling.
# That means if we measure with daily return, K is sqrt(250)
# scatter plot the data
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
start = '2017-04-22'
end = '2018-04-22'

from pandas_datareader import data, wb
import pandas_datareader as pdr
pdr.get_data_yahoo('AAPL', start=start, end=end)  # if by default, returns 2009-2019, 2403 rows, daily stock data
pdr.get_data_quandl('EOD/BA', start=start, end=end, api_key='DGCrDAbR7GMbx7v9wSCW')

def get_risk(prices):
    return (prices / prices.shift(1) - 1).dropna().std().values


def get_return(prices):
    return ((prices / prices.shift(1) - 1).dropna().mean() * np.sqrt(250)).values


# symbols = ['BA', 'AAPL', 'BRK_A', 'MSFT']
symbols = ['BA', 'C', 'AAL', 'NFLX']
prices = pd.DataFrame(index=pd.date_range(start, end))
for symbol in symbols:
    # portfolio = web.DataReader(name=symbol, data_source='quandl', start=start, end=end, 'DGCrDAbR7GMbx7v9wSCW')
    # portfolio = pdr.get_data_quandl('EOD/'+symbol, start=start, end=end, api_key='DGCrDAbR7GMbx7v9wSCW')
    portfolio = pdr.get_data_yahoo(symbol, start=start, end=end)
    close = portfolio[['Adj Close']]
    close = close.rename(columns={'Adj Close': symbol})
    prices = prices.join(close)  # join by datetime index
    portfolio.to_csv("~/workspace/{}.csv".format(symbol))

prices = prices.dropna()
risk_v = get_risk(prices)
return_v = get_return(prices)
fig, ax = plt.subplots()
ax.scatter(x=risk_v, y=return_v, alpha=0.5)
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
plt.show()

########################################################################
# Build portfolio with random weights


def random_weights(n):
    weights = np.random.rand(n)
    return weights / sum(weights)


def get_portfolio_risk(weights, normalized_prices):
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=normalized_prices.index, data={'portfolio': portfolio_val})
    return (portfolio / portfolio.shift(1) - 1).dropna().std().values[0]


def get_portfolio_return(weights, normalized_prices):
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=normalized_prices.index, data={'portfolio': portfolio_val})
    ret = get_return(portfolio)
    return ret[0]


risk_all = np.array([])
return_all = np.array([])
# for demo purpose, plot 3000 random portoflio
np.random.seed(0)
normalized_prices = prices / prices.ix[0, :]
for _ in range(0, 3000):
    weights = random_weights(len(symbols))
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=prices.index, data={'portfolio': portfolio_val})
    risk = get_risk(portfolio)
    ret = get_return(portfolio)
    risk_all = np.append(risk_all, risk)
    return_all = np.append(return_all, ret)
    p = get_portfolio_risk(weights=weights, normalized_prices=normalized_prices)

fig, ax = plt.subplots()
ax.scatter(x=risk_all, y=return_all, alpha=0.5)
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')

for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))

ax.scatter(x=risk_v, y=return_v, alpha=0.5, color='red')
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')
ax.grid()
plt.show()

##################################################################################
# scipy’s optimizer to get optimal weights for different targeted return
# weights in range [0, 1] & sum of weights is 1
# portfolio return = target return
# scipy optimizer is able to find the best allocation
# optimizer


def optimize(prices, symbols, target_return=0.1):
    normalized_prices = prices / prices.ix[0, :]
    init_guess = np.ones(len(symbols)) * (1.0 / len(symbols))
    bounds = ((0.0, 1.0),) * len(symbols)
    weights = minimize(get_portfolio_risk, init_guess,
                       args=(normalized_prices,), method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
                                    {'type': 'eq', 'args': (normalized_prices,),
                                     'fun': lambda inputs, normalized_prices:
                                     target_return - get_portfolio_return(weights=inputs,
                                                                          normalized_prices=normalized_prices)}),
                       bounds=bounds)
    return weights.x
optimal_risk_all = np.array([])
optimal_return_all = np.array([])
for target_return in np.arange(0.005, .0402, .0005):
    opt_w = optimize(prices=prices, symbols=symbols, target_return=target_return)
    optimal_risk_all = np.append(optimal_risk_all, get_portfolio_risk(opt_w, normalized_prices))
    optimal_return_all = np.append(optimal_return_all, get_portfolio_return(opt_w, normalized_prices))
# plotting
fig, ax = plt.subplots()
# random portfolio risk return
ax.scatter(x=risk_all, y=return_all, alpha=0.5)
# optimal portfolio risk return
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
ax.plot(optimal_risk_all, optimal_return_all, '-', color='green')
# symbol risk return
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
ax.scatter(x=risk_v, y=return_v, color='red')
ax.set(title='Efficient Frontier', xlabel='Risk', ylabel='Return')
ax.grid()
plt.savefig('return_risk_efficient_frontier.png', bbox_inches='tight')

####################################################################################
# what is the best allocation overall? Portfolio performance can be
# evaluated with return/risk ratio (known as Sharpe Ratio)
































































