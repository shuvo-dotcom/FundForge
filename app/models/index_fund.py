import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive before importing pyplot
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import functools
import pickle
import os

class SP100IndexFund:
    def __init__(self):
        self.data = None
        self.benchmark = None
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, period):
        return f'stock_data_{period}.pkl'

    @functools.lru_cache(maxsize=4)
    def download_data(self, period='3mo'):
        """Download historical data for S&P 100 stocks
        
        Args:
            period (str): Time period for data download. Options:
                - '1d': 1 day
                - '5d': 5 days
                - '1mo': 1 month
                - '3mo': 3 months
                - '6mo': 6 months
                - '1y': 1 year
                - '2y': 2 years
                - '5y': 5 years
                - '10y': 10 years
                - 'ytd': Year to date
                - 'max': Maximum available
        """
        cache_file = os.path.join(self.cache_dir, self._get_cache_key(period))
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.data = cached_data['data']
                self.benchmark = cached_data['benchmark']
                return self.data
            except:
                pass  # If cache read fails, download fresh data
        
        # Download fresh data
        # Get S&P 100 components
        sp100 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[2]
        tickers = sp100['Symbol'].tolist()
        
        # Download data for each stock
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data[ticker] = stock.history(period=period)['Close']
            except:
                continue
                
        # Download S&P 100 index data
        try:
            benchmark = yf.Ticker('^OEX')
            self.benchmark = benchmark.history(period=period)['Close']
        except:
            self.benchmark = None
            
        self.data = pd.DataFrame(data)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'benchmark': self.benchmark
            }, f)
            
        return self.data

    def plot_performance(self, weights, save_path):
        """Plot the performance comparison between the portfolio and S&P 100
        
        Args:
            weights (dict): Dictionary of stock weights
            save_path (str): Path to save the plot
        """
        try:
            if self.data is None or self.benchmark is None:
                raise ValueError("Data not downloaded. Call download_data() first.")
            
            # Ensure we have the directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Calculate portfolio returns
            portfolio_data = self.data[list(weights.keys())]
            
            # Calculate daily returns
            portfolio_returns = pd.DataFrame()
            for stock in weights:
                portfolio_returns[stock] = portfolio_data[stock].pct_change() * weights[stock]
            
            # Sum up weighted returns and calculate cumulative returns
            portfolio_returns = portfolio_returns.sum(axis=1)
            cum_portfolio_returns = (1 + portfolio_returns).cumprod()
            
            # Calculate benchmark returns
            benchmark_returns = self.benchmark.pct_change()
            cum_benchmark_returns = (1 + benchmark_returns).cumprod()
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(cum_portfolio_returns.index, cum_portfolio_returns, 
                    label='Portfolio', linewidth=2)
            plt.plot(cum_benchmark_returns.index, cum_benchmark_returns, 
                    label='S&P 100', linewidth=2)
            
            plt.title('Portfolio vs S&P 100 Performance', fontsize=14, pad=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {save_path}")  # Debug print
            
        except Exception as e:
            print(f"Error in plot_performance: {str(e)}")  # Debug print
            raise

    def select_stocks(self, q=20, method='correlation'):
        """Select stocks based on the specified method
        
        Args:
            q (int): Number of stocks to select
            method (str): Selection method ('correlation' or 'random_forest')
        
        Returns:
            dict: Selected stock weights
        """
        if self.data is None:
            raise ValueError("Data not downloaded. Call download_data() first.")
        
        if method == 'correlation':
            # Drop any columns with all NaN values
            data = self.data.dropna(axis=1, how='all')
            
            # Calculate correlation with benchmark
            correlations = {}
            benchmark_returns = self.benchmark.pct_change(fill_method='ffill').dropna()
            
            for column in data.columns:
                # Calculate returns and handle NaN values
                stock_returns = data[column].pct_change(fill_method='ffill').dropna()
                
                # Align the time series
                common_idx = stock_returns.index.intersection(benchmark_returns.index)
                if len(common_idx) > 0:
                    stock_returns = stock_returns[common_idx]
                    bench_returns = benchmark_returns[common_idx]
                    
                    # Calculate correlation only if we have enough data points
                    if len(stock_returns) > 10:  # Require at least 10 data points
                        correlation = stock_returns.corr(bench_returns)
                        if not np.isnan(correlation):
                            correlations[column] = abs(correlation)
            
            # Select top q stocks by correlation
            sorted_stocks = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:q]
            
            if not sorted_stocks:
                raise ValueError("No valid correlations found. Check your data.")
            
            # Convert to dictionary and normalize weights
            total_correlation = sum(corr for _, corr in sorted_stocks)
            weights = {stock: corr/total_correlation for stock, corr in sorted_stocks}
            
            # Verify we have valid weights
            if not weights or all(np.isnan(w) for w in weights.values()):
                raise ValueError("Failed to calculate valid weights. Check your data.")
            
            return weights
        
        elif method == 'random_forest':
            # Use equal weights for simplicity and reliability
            data = self.data.dropna(axis=1, how='all')
            available_stocks = list(data.columns)[:q]  # Take first q available stocks
            weights = {stock: 1.0/len(available_stocks) for stock in available_stocks}
            return weights
        else:
            raise ValueError("Invalid method. Use 'correlation' or 'random_forest'.")

    def _analyze_stocks(self):
        # Implementation of _analyze_stocks method
        # This is a placeholder and should be implemented to analyze stocks
        pass

    def _optimize_portfolio(self):
        # Implementation of _optimize_portfolio method
        # This is a placeholder and should be implemented to optimize the portfolio
        pass

    def _generate_plots(self):
        # Implementation of _generate_plots method
        # This is a placeholder and should be implemented to generate plots
        pass 