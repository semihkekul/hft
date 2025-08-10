import numpy as np
from scipy.stats import norm


class Util:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    @staticmethod
    def simulate_stock_prices(S0, mu, sigma, T, dt):
        N = int(T / dt)
        t = np.arange(N)  # time as trading days
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        return t, S



class Model:
    @staticmethod
    def simple(market_prices, theoretical_prices, threshold):
        position = 0
        cash = 0
        positions = []
        profits = []
        signals = []
        
        for i in range(len(market_prices)):
    
            if market_prices[i] < theoretical_prices[i] - threshold:
                position += 1
                cash -= market_prices[i]
                signals.append('Buy')
            elif market_prices[i] > theoretical_prices[i] + threshold and position > 0:
                position -= 1
                cash += market_prices[i]
                signals.append('Sell')
            else:
                signals.append(None)

            positions.append(position)
            profits.append(cash + position * market_prices[i])      

        return profits, signals    
            
    @staticmethod
    def model2():
        pass

    @staticmethod
    def model3():
        pass
    