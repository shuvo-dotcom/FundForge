from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt

# Create blueprint without url_prefix to make it the main/index page
index_fund_bp = Blueprint('index_fund', __name__)

@index_fund_bp.route('/', methods=['GET'])
def index():
    min_stocks = int(os.getenv('MIN_STOCKS', 10))
    max_stocks = int(os.getenv('MAX_STOCKS', 100))
    default_stocks = int(os.getenv('DEFAULT_NUM_STOCKS', 20))
    
    return render_template('index_fund/index.html',
                         min_stocks=min_stocks,
                         max_stocks=max_stocks,
                         default_stocks=default_stocks)

@index_fund_bp.route('/analyze', methods=['POST'])
def analyze():
    try:
        min_stocks = int(os.getenv('MIN_STOCKS', 10))
        max_stocks = int(os.getenv('MAX_STOCKS', 100))
        
        q = int(request.form.get('q', 0))
        if not min_stocks <= q <= max_stocks:
            flash(f'Please enter a number between {min_stocks} and {max_stocks}.', 'error')
            return redirect(url_for('index_fund.index'))

        # Initialize
        index_fund = SP100IndexFund()
        
        # Download data
        try:
            index_fund.download_data(period='3mo')
        except Exception as e:
            flash(f'Error downloading data: {str(e)}', 'error')
            return redirect(url_for('index_fund.index'))
        
        # Select stocks
        try:
            selected_stocks = index_fund.select_stocks(q=q, method='correlation')
            if not selected_stocks:
                raise ValueError("No stocks were selected")
        except Exception as e:
            flash(f'Error selecting stocks: {str(e)}', 'error')
            return redirect(url_for('index_fund.index'))
        
        # Generate plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        static_folder = current_app.static_folder
        plots_dir = os.path.join(static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_filename = f'performance_{timestamp}.png'
        plot_path = os.path.join(plots_dir, plot_filename)
        relative_plot_path = f'plots/{plot_filename}'
        
        try:
            index_fund.plot_performance(selected_stocks, plot_path)
            print(f"Plot saved to: {plot_path}")  # Debug print
        except Exception as e:
            print(f"Error generating plot: {str(e)}")  # Debug print
            flash(f'Error generating plot: {str(e)}', 'error')
            relative_plot_path = None
        
        return render_template('index_fund/results.html',
                             plot_path=relative_plot_path,
                             selected_stocks=selected_stocks)
                             
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index_fund.index'))

# Add SP100IndexFund class here if not imported from elsewhere
class SP100IndexFund:
    def __init__(self):
        self.data = None
        self.benchmark = None
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_data(self, period='3mo'):
        """Download historical data for S&P 100 stocks"""
        # Get S&P 100 components
        try:
            sp100 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[2]
            tickers = sp100['Symbol'].tolist()
        except Exception as e:
            print(f"Error getting S&P 100 components: {str(e)}")
            return None
        
        # Download data for each stock
        data = {}
        for ticker in tickers:
            try:
                # Handle special cases
                if ticker == 'BRK.B':
                    ticker = 'BRK-B'  # Try alternative ticker format
                
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker] = hist['Close']
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
                continue
                
        self.data = pd.DataFrame(data)
        
        # Download S&P 100 index data
        try:
            benchmark = yf.Ticker('^OEX')
            self.benchmark = benchmark.history(period=period)['Close']
        except Exception as e:
            print(f"Error downloading benchmark data: {str(e)}")
            return None
        
        return self.data

    def select_stocks(self, q=20, method='correlation'):
        """Select stocks based on correlation with benchmark"""
        if self.data is None or self.benchmark is None:
            raise ValueError("Data not downloaded. Call download_data() first.")
            
        # Calculate correlation with benchmark
        correlations = {}
        # Fill NA values before calculating returns
        filled_data = self.data.fillna(method='ffill')
        filled_benchmark = self.benchmark.fillna(method='ffill')
        
        for column in filled_data.columns:
            stock_returns = filled_data[column].pct_change().dropna()
            benchmark_returns = filled_benchmark.pct_change().dropna()
            
            # Align the time series
            common_idx = stock_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 0:
                stock_returns = stock_returns[common_idx]
                bench_returns = benchmark_returns[common_idx]
                correlation = stock_returns.corr(bench_returns)
                if not np.isnan(correlation):
                    correlations[column] = abs(correlation)
        
        # Select top q stocks
        sorted_stocks = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:q]
        
        # Convert to dictionary and normalize weights
        total_correlation = sum(corr for _, corr in sorted_stocks)
        weights = {stock: corr/total_correlation for stock, corr in sorted_stocks}
        
        return weights

    def plot_performance(self, weights, save_path):
        """Plot performance comparison"""
        if self.data is None or self.benchmark is None:
            raise ValueError("Data not downloaded. Call download_data() first.")
            
        # Fill NA values before calculating returns
        portfolio_data = self.data[list(weights.keys())].fillna(method='ffill')
        filled_benchmark = self.benchmark.fillna(method='ffill')
        
        # Calculate portfolio returns
        weighted_returns = portfolio_data.pct_change() * pd.Series(weights)
        portfolio_returns = weighted_returns.sum(axis=1)
        cum_portfolio_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate benchmark returns
        benchmark_returns = filled_benchmark.pct_change()
        cum_benchmark_returns = (1 + benchmark_returns).cumprod()
        
        # Plot
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
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Ensure the directory exists and save plot
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 