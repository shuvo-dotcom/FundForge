import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import subprocess
import tempfile
import seaborn as sns
from .ai_optimizer import AIOptimizer

class SP100IndexFund:
    def __init__(self):
        self.sp100_tickers = self._get_sp100_tickers()
        self.data = None
        self.benchmark = None
        self.ai_optimizer = AIOptimizer()
    
    def _get_sp100_tickers(self):
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'JPM', 'JNJ', 'V', 'PG',
            'NVDA', 'HD', 'MA', 'BAC', 'DIS', 'PFE', 'KO', 'NFLX', 'MRK', 'PEP',
            'ABBV', 'TMO', 'CSCO', 'WMT', 'MCD', 'ABT', 'CVX', 'ACN', 'AVGO', 'T',
            'CRM', 'COST', 'NKE', 'VZ', 'DHR', 'UNH', 'LIN', 'PM', 'BMY', 'LOW',
            'HON', 'ORCL', 'UPS', 'SBUX', 'AMGN', 'TXN', 'QCOM', 'INTU', 'IBM', 'GS',
            'UNP', 'CAT', 'MMM', 'AXP', 'GE', 'BA', 'RTX', 'PYPL', 'MDT', 'CMCSA',
            'BLK', 'LMT', 'MO', 'ADP', 'GILD', 'DE', 'ISRG', 'AMAT', 'SCHW', 'BKNG',
            'PLD', 'CI', 'SYK', 'MDLZ', 'ADI', 'TJX', 'NOW', 'ZTS', 'BDX', 'MMC',
            'REGN', 'TGT', 'LRCX', 'FIS', 'DUK', 'SO', 'NEE', 'D', 'CL', 'EL',
            'AON', 'PGR', 'ICE', 'SPGI', 'CCI', 'APD', 'SHW', 'ECL', 'ROP', 'VRTX'
        ]
    
    def download_data(self, period='1y'):
        print("Downloading historical data...")
        data = {}
        for ticker in self.sp100_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker] = hist['Close']
                else:
                    print(f"No data available for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        try:
            self.benchmark = yf.Ticker('^OEX').history(period=period)['Close']
        except Exception as e:
            print(f"Error downloading S&P 100 index: {e}")
        
        self.data = pd.DataFrame(data)
        print("Data download completed.")
    
    def select_stocks(self, q=20, method='ai'):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data.pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        if method == 'ai':
            # Use AI optimizer for stock selection
            weights, selected_stocks = self.ai_optimizer.optimize_portfolio(
                returns, benchmark_returns, q=q
            )
            if X.empty or y.empty:
                raise ValueError("No data available for training the model.")
            return selected_stocks
        else:
            # Fallback to correlation-based selection
            correlations = returns.apply(lambda x: x.corr(benchmark_returns))
            correlations = correlations.fillna(0)
            selected_stocks = list(correlations.nlargest(q).index)
            return selected_stocks
    
    def optimize_weights(self, selected_stocks, method='ai'):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data[selected_stocks].pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        if method == 'ai':
            # Use AI optimizer for weight optimization
            weights, _ = self.ai_optimizer.optimize_portfolio(
                returns, benchmark_returns, q=len(selected_stocks)
            )
            return weights
        else:
            # Fallback to mean-variance optimization
            n = len(selected_stocks)
            weights = np.ones(n) / n  # Equal weights as initial guess
            
            def objective(weights):
                portfolio_returns = returns.dot(weights)
                tracking_error = np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))
                return tracking_error
            
            from scipy.optimize import minimize
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                {'type': 'ineq', 'fun': lambda w: w}  # Weights non-negative
            ]
            
            result = minimize(objective, weights, constraints=constraints, method='SLSQP')
            
            # Create weights dictionary
            weights_dict = {stock: weight for stock, weight in zip(selected_stocks, result.x)}
            return weights_dict
    
    def calculate_correlation(self, weights):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data[list(weights.keys())].pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock, weight in weights.items():
            portfolio_returns += weight * returns[stock]
        
        return portfolio_returns.corr(benchmark_returns)
    
    def calculate_tracking_error(self, weights):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data[list(weights.keys())].pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock, weight in weights.items():
            portfolio_returns += weight * returns[stock]
        
        return np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))
    
    def calculate_excess_return(self, weights):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data[list(weights.keys())].pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock, weight in weights.items():
            portfolio_returns += weight * returns[stock]
        
        return portfolio_returns.mean() - benchmark_returns.mean()
    
    def plot_performance(self, weights, save_path):
        if self.data is None:
            raise ValueError("Please download data first using download_data()")
        
        returns = self.data[list(weights.keys())].pct_change(fill_method=None).dropna()
        benchmark_returns = self.benchmark.pct_change(fill_method=None).dropna()
        
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock, weight in weights.items():
            portfolio_returns += weight * returns[stock]
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns.cumsum(), label='Portfolio')
        plt.plot(benchmark_returns.cumsum(), label='S&P 100')
        plt.title('Portfolio vs S&P 100 Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        plt.close()

def main():
    # Initialize the index fund
    fund = SP100IndexFund()
    
    # Download data
    fund.download_data()
    
    # Test different selection methods
    methods = {
        'Correlation-Based': fund.select_stocks,
        'Sector-Based': fund.select_stocks,
        'Momentum-Based': fund.select_stocks,
        'ML-Based': fund.select_stocks
    }
    
    # Test different values of q
    q_values = [10, 20, 30, 40]
    
    results = {}
    for method_name, method in methods.items():
        results[method_name] = {}
        for q in q_values:
            print(f"\nTesting {method_name} with q={q}")
            selected_stocks = method(q=q)
            
            # Optimize portfolio weights
            weights = fund.optimize_weights(selected_stocks)
            results[method_name][q] = weights
            
            # Plot results
            fund.plot_performance(weights, f"{method_name} (q={q})")
            
            print(f"Correlation: {fund.calculate_correlation(weights):.4f}")
            print(f"Tracking Error: {fund.calculate_tracking_error(weights):.4f}")
            print(f"Excess Return: {fund.calculate_excess_return(weights):.4f}")
            print("Top 5 weights:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for stock, weight in sorted_weights[:5]:
                print(f"  {stock}: {weight:.2%}")

if __name__ == "__main__":
    main() 