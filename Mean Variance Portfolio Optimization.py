 ##### Stock Analyser #####

##### Imported Modules #####
import pandas_datareader as wb
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

### Time interval and company ticker###
start = datetime(2010,1,1)
mid = datetime(2015,1,1)
end = datetime(2020,1,1)
source = 'yahoo'
tickers = ['GS','AAPL','MSFT','wmt']


### stock dataset ###
df = wb.DataReader(tickers, source, start, end)['Adj Close']
df.reset_index(inplace=True,drop=False)

one_year = df[df['Date'].dt.year ==2019].reset_index()
five_year = df[(df['Date']>=mid)& (df['Date']<=end)].reset_index()
ten_year = df[(df['Date']>=start) & (df['Date']<=end)].reset_index()

del one_year['Date']
del one_year['index']
del five_year['Date']
del five_year['index']
del ten_year['Date']
del ten_year['index']


### Mean Variance Portfolio Optimization ### 
investment_horizon = one_year
mean_returns = investment_horizon.pct_change().mean()
cov = investment_horizon.pct_change().cov()
num_portfolios = 1000
rf = 0.0062

### The function below is called by the function that is below it again. 
def calc_portfolio(weights, mean_returns, cov, rf):
	portfolio_return = np.sum(mean_returns*weights)*252
	portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov,weights)))*np.sqrt(252)
	sharpe_ratio = (portfolio_return - rf) / portfolio_std
	return portfolio_return, portfolio_std, sharpe_ratio

### Here we find the weights ###
def simulate_portfolios(num_portfolios, mean_returns, cov, rf):
	returns_matrix = np.zeros((len(mean_returns)+3,num_portfolios))
	for i in range(num_portfolios):
		weights = np.random.random(len(mean_returns))
		weights /= np.sum(weights)
		portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio(weights, mean_returns, cov, rf)
		returns_matrix[0,i] = portfolio_return
		returns_matrix[1,i] = portfolio_std
		returns_matrix[2,i] = sharpe_ratio
		for j in range(len(weights)):
			returns_matrix[j+3,i] = weights[j]
	results_df = pd.DataFrame(returns_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
	results_df['ret'] = results_df['ret'].apply(lambda x: x*100)
	results_df['stdev'] = results_df['stdev'].apply(lambda x: x*100)
	return results_df


### Results frame is the dataframe with all the portfolio weights, sharpe ratios, returns, and standard deviations
results_frame = simulate_portfolios(num_portfolios, mean_returns, cov, rf)

### Below is the two variables with the max sharpe and min volatility 
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

#Below is just the plot for the frontier and which portfolios are the most efficient 
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation %')
plt.ylabel('Returns %')
plt.colorbar()
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
#plt.show()

### Here are the most efficient weights for each stock in our portfolio ###
print('------------------------------------------------------------------')
print('Max sharpe ratio:')
print(max_sharpe_port.to_frame().T.reset_index())
print('------------------------------------------------------------------')
print('Minimum volatility')
print(min_vol_port.to_frame().T.reset_index())
