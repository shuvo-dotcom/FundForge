import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

class SP100IndexFund:
    def __init__(self):
        self.sp100_tickers = self._get_sp100_tickers()
        self.cache_dir = os.getenv('CACHE_DIR', 'app/cache')
        self.plots_dir = os.getenv('PLOTS_DIR', 'static/plots')
        self.etf_symbol = os.getenv('SP100_ETF_SYMBOL', '^OEX')
        self.min_stocks = int(os.getenv('MIN_STOCKS', 10))
        self.max_stocks = int(os.getenv('MAX_STOCKS', 100))
        
    def _get_sp100_tickers(self):
        # For demonstration, using a subset of S&P 100 stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'UNH',
                'MA', 'HD', 'BAC', 'XOM', 'PFE', 'DIS', 'CSCO', 'VZ', 'CMCSA', 'ADBE']
    
    def _get_cache_path(self, period):
        return os.path.join(self.cache_dir, f'data_{period}.pkl')
    
    def download_data(self, period='1y'):
        cache_path = self._get_cache_path(period)
        
        # Check if cached data exists and is not too old (1 day)
        if os.path.exists(cache_path):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if cache_age.days < 1:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.benchmark = data['benchmark']
                    self.stock_data = data['stock_data']
                    self.benchmark_returns = data['benchmark_returns']
                    self.stock_returns = data['stock_returns']
                    return
        
        end_date = datetime.now()
        if period == '3mo':
            start_date = end_date - timedelta(days=63)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '9mo':
            start_date = end_date - timedelta(days=270)
        else:  # 1y
            start_date = end_date - timedelta(days=365)
        
        # Download S&P 100 ETF data as benchmark
        self.benchmark = yf.download(self.etf_symbol, start=start_date, end=end_date)['Adj Close']
        
        # Download individual stock data
        self.stock_data = pd.DataFrame()
        for ticker in self.sp100_tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                self.stock_data[ticker] = data
            except:
                continue
        
        # Calculate daily returns
        self.benchmark_returns = self.benchmark.pct_change().dropna()
        self.stock_returns = self.stock_data.pct_change().dropna()
        
        # Cache the data
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'benchmark': self.benchmark,
                'stock_data': self.stock_data,
                'benchmark_returns': self.benchmark_returns,
                'stock_returns': self.stock_returns
            }, f)
    
    def select_stocks(self, q=20, method='correlation'):
        if not self.min_stocks <= q <= self.max_stocks:
            raise ValueError(f'Number of stocks must be between {self.min_stocks} and {self.max_stocks}')
            
        if method == 'correlation':
            # Select stocks based on correlation with benchmark
            correlations = self.stock_returns.corrwith(self.benchmark_returns)
            return correlations.nlargest(q).index.tolist()
        elif method == 'random_forest':
            # Select stocks using Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(self.stock_returns, self.benchmark_returns)
            importance = pd.Series(rf.feature_importances_, index=self.stock_returns.columns)
            return importance.nlargest(q).index.tolist()
    
    def optimize_weights(self, selected_stocks, method='linear_regression'):
        selected_returns = self.stock_returns[selected_stocks]
        
        if method == 'linear_regression':
            # Use linear regression to find optimal weights
            scaler = StandardScaler()
            X = scaler.fit_transform(selected_returns)
            y = self.benchmark_returns.values
            
            model = LinearRegression()
            model.fit(X, y)
            weights = model.coef_
            weights = np.maximum(weights, 0)  # Ensure non-negative weights
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
        elif method == 'equal_weight':
            # Equal weighting
            weights = np.ones(len(selected_stocks)) / len(selected_stocks)
            
        return dict(zip(selected_stocks, weights))
    
    def calculate_metrics(self, weights):
        selected_stocks = list(weights.keys())
        selected_returns = self.stock_returns[selected_stocks]
        weights_array = np.array(list(weights.values()))
        
        # Calculate portfolio returns
        portfolio_returns = (selected_returns * weights_array).sum(axis=1)
        
        # Calculate metrics
        correlation = np.corrcoef(portfolio_returns, self.benchmark_returns)[0, 1]
        tracking_error = np.std(portfolio_returns - self.benchmark_returns) * np.sqrt(252) * 100
        excess_return = (portfolio_returns.mean() - self.benchmark_returns.mean()) * 252 * 100
        
        return {
            'correlation': correlation,
            'tracking_error': tracking_error,
            'excess_return': excess_return,
            'portfolio_returns': portfolio_returns
        }
    
    def plot_performance(self, weights, save_path=None):
        selected_stocks = list(weights.keys())
        selected_returns = self.stock_returns[selected_stocks]
        weights_array = np.array(list(weights.values()))
        
        # Calculate portfolio returns
        portfolio_returns = (selected_returns * weights_array).sum(axis=1)
        
        # Create performance plot
        plt.figure(figsize=(10, 6))
        plt.plot((1 + self.benchmark_returns).cumprod(), label='S&P 100', color='blue')
        plt.plot((1 + portfolio_returns).cumprod(), label='Index Fund', color='green')
        plt.title('Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 