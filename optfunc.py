import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

def rand_portfolios(expected_returns, cov_matrix):
    
    # Declare random portfolios properties
    port_returns = []
    port_volatility = []
    stock_weights = []

    selected = (expected_returns.axes)[0]

    num_assets = len(selected)
    num_portfolios = 100000
    
    # Create random portfolios
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}

    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]

    df = pd.DataFrame(portfolio)

    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]

    df = df[column_order]

    return df


def optimal_portfolios(returns):
    
    n = returns.shape[1]
    returns = np.transpose(returns.as_matrix())

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Compute efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    
    # Compute risks and returns for frontier
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    
    # Compute 2nd degree polynomial of frontier curve
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    
    # Compute optimal portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    
    return np.asarray(wt), returns, risks
