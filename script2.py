import pandas as pd
import matplotlib.pyplot as plt
from optfunc import rand_portfolios, optimal_portfolios
import numpy as np

def main():

     # Load stock data
    stock_data = pd.read_csv('stock_price_data.csv')
    print(stock_data)

    # Calculate monthly return for each period
    selected = list(stock_data.columns[1:])
    returns_monthly = stock_data[selected].pct_change()
    print('returns_monthly\n')
    print(returns_monthly)

    # Calculate expected returns
    expected_returns = returns_monthly.mean()

    # Calculate covariance
    cov_monthly = returns_monthly.cov()

    # Find set of random portfolios
    random_portfolios = rand_portfolios(expected_returns, cov_monthly)

    # Plot set of random portfolios
    random_portfolios.plot.scatter(x='Volatility', y='Returns', fontsize=12, color='c', label='random portfolio')
    
    # Calculate set of portfolios on the EF
    weights, returns, risks = optimal_portfolios(returns_monthly[1:])

    # Plot set of portfolios on the EF
    plt.plot(risks, returns, 'b-o', label='efficient portfolio')
    plt.ylabel('Expected Returns',fontsize=14)
    plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
    plt.title('Efficient Frontier Top10 US Cap', fontsize=24)
 
    # Compare set of portfolios on the EF to the single stock
    try:
      single_asset_std=np.sqrt(np.diagonal(cov_monthly))
      plt.scatter(single_asset_std,expected_returns,marker='*',color='red',s=200, label='single stock')
      plt.legend()
    except:
      pass
    plt.show()

if __name__ == '__main__':
    main()
