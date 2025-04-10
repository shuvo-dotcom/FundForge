import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class SP100IndexFund:
    def __init__(self):
        self.sp100_tickers = self._get_sp100_tickers()
        
    def _get_sp100_tickers(self):
        # For demonstration, using a subset of S&P 100 stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'UNH',
                'MA', 'HD', 'BAC', 'XOM', 'PFE', 'DIS', 'CSCO', 'VZ', 'CMCSA', 'ADBE']
    
    def analyze(self, num_stocks):
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download S&P 100 ETF data as benchmark
        benchmark = yf.download('^OEX', start=start_date, end=end_date)['Adj Close']
        
        # Download individual stock data
        stock_data = pd.DataFrame()
        for ticker in self.sp100_tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                stock_data[ticker] = data
            except:
                continue
        
        # Calculate daily returns
        benchmark_returns = benchmark.pct_change().dropna()
        stock_returns = stock_data.pct_change().dropna()
        
        # Calculate correlations with benchmark
        correlations = stock_returns.corrwith(benchmark_returns)
        selected_stocks = correlations.nlargest(num_stocks).index.tolist()
        
        # Calculate optimal weights using simple linear regression
        selected_returns = stock_returns[selected_stocks]
        scaler = StandardScaler()
        X = scaler.fit_transform(selected_returns)
        y = benchmark_returns.values
        
        model = LinearRegression()
        model.fit(X, y)
        weights = model.coef_
        weights = np.maximum(weights, 0)  # Ensure non-negative weights
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Calculate portfolio returns
        portfolio_returns = (selected_returns * weights).sum(axis=1)
        
        # Calculate metrics
        correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252) * 100
        excess_return = (portfolio_returns.mean() - benchmark_returns.mean()) * 252 * 100
        
        # Create performance plot
        plt.figure(figsize=(10, 6))
        plt.plot((1 + benchmark_returns).cumprod(), label='S&P 100', color='blue')
        plt.plot((1 + portfolio_returns).cumprod(), label='Index Fund', color='green')
        plt.title('Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f'plots/performance_{timestamp}.png'
        os.makedirs('static/plots', exist_ok=True)
        plt.savefig(f'static/{plot_path}')
        plt.close()
        
        return {
            'correlation': correlation,
            'tracking_error': tracking_error,
            'excess_return': excess_return,
            'selected_stocks': list(zip(selected_stocks, weights)),
            'plot_path': plot_path
        } 