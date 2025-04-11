import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import requests
from datetime import datetime, timedelta

class SP100IndexFund:
    def __init__(self):
        self.data = None
        self.benchmark = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_stock_data(self, ticker, period='3mo'):
        """Get stock data directly from Yahoo Finance API"""
        try:
            # Convert period to days
            if period == '3mo':
                days = 90
            elif period == '6mo':
                days = 180
            elif period == '9mo':
                days = 270
            elif period == '1y':
                days = 365
            else:
                days = 90

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Format dates for Yahoo Finance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Construct URL
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}&interval=1d'

            # Make request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Extract price data
            timestamps = data['chart']['result'][0]['timestamp']
            prices = data['chart']['result'][0]['indicators']['quote'][0]['close']

            # Create DataFrame
            df = pd.DataFrame({
                'Close': prices
            }, index=pd.to_datetime(timestamps, unit='s'))

            return df['Close']

        except Exception as e:
            st.warning(f"Error fetching {ticker}: {str(e)}")
            return None

    def download_data(self, period='3mo'):
        """Download S&P 100 data with improved error handling"""
        with st.spinner('Downloading S&P 100 data...'):
            try:
                # Get S&P 100 components
                sp100_url = 'https://en.wikipedia.org/wiki/S%26P_100'
                response = requests.get(sp100_url, headers=self.headers)
                sp100 = pd.read_html(response.text)[2]
                tickers = sp100['Symbol'].tolist()

                data = {}
                progress_bar = st.progress(0)

                # Download stock data
                for i, ticker in enumerate(tickers):
                    try:
                        if ticker == 'BRK.B':
                            ticker = 'BRK-B'
                        
                        # Add delay between requests
                        time.sleep(0.5)
                        
                        stock_data = self.get_stock_data(ticker, period)
                        if stock_data is not None and not stock_data.empty:
                            data[ticker] = stock_data
                            
                    except Exception as e:
                        st.warning(f"Error processing {ticker}: {str(e)}")
                    finally:
                        progress_bar.progress((i + 1) / len(tickers))

                if not data:
                    st.error("Failed to download any stock data")
                    return None

                self.data = pd.DataFrame(data)

                # Download benchmark data
                time.sleep(1)  # Wait before benchmark request
                benchmark_data = self.get_stock_data('^OEX', period)
                if benchmark_data is not None and not benchmark_data.empty:
                    self.benchmark = benchmark_data
                else:
                    st.error("Failed to get benchmark data")
                    return None

                return self.data

            except Exception as e:
                st.error(f"Error in download_data: {str(e)}")
                return None

    def optimize_portfolio(self, q=20):
        """Mathematical optimization approach using scipy"""
        if self.data is None or self.benchmark is None:
            raise ValueError("Data not downloaded")

        # Calculate returns
        returns = self.data.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()

        # Align dates
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

        n_stocks = len(returns.columns)

        def objective(weights):
            """Minimize negative correlation (maximize correlation)"""
            portfolio_returns = np.dot(returns, weights)
            return -np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]

        def constraint_sum(weights):
            """Weights must sum to 1"""
            return np.sum(weights) - 1

        def constraint_num_stocks(weights):
            """Number of non-zero weights should be q"""
            return np.sum(weights > 0.001) - q

        # Initial guess (equal weights for first q stocks)
        x0 = np.zeros(n_stocks)
        x0[:q] = 1/q

        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n_stocks)]

        # Optimize
        with st.spinner("Optimizing portfolio weights..."):
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': constraint_sum},
                    {'type': 'eq', 'fun': constraint_num_stocks}
                ],
                options={'maxiter': 1000}
            )

        # Convert results to dictionary
        weights = {}
        for i, weight in enumerate(result.x):
            if weight > 0.001:  # Filter small weights
                weights[returns.columns[i]] = weight

        return weights

    def calculate_metrics(self, weights):
        """Calculate performance metrics"""
        returns = self.data[list(weights.keys())].pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock, weight in weights.items():
            portfolio_returns += weight * returns[stock]
        
        # Calculate metrics
        correlation = portfolio_returns.corr(benchmark_returns)
        tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
        excess_return = (portfolio_returns.mean() - benchmark_returns.mean()) * 252
        
        return {
            'correlation': correlation,
            'tracking_error': tracking_error,
            'excess_return': excess_return,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns
        }

    def analyze_q_vs_correlation(self, q_range=range(10, 101, 10), period='3mo'):
        """Analyze correlation for different values of q"""
        correlations = []
        q_values = []
        
        with st.spinner('Analyzing different portfolio sizes...'):
            for q in q_range:
                try:
                    # Optimize portfolio for this q
                    weights = self.optimize_portfolio(q)
                    metrics = self.calculate_metrics(weights)
                    correlations.append(metrics['correlation'])
                    q_values.append(q)
                    
                    # Show progress
                    st.write(f"q={q}: Correlation = {metrics['correlation']:.4f}")
                    
                except Exception as e:
                    st.warning(f"Error at q={q}: {str(e)}")
                    continue
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(q_values, correlations, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Stocks (q)', fontsize=12)
        ax.set_ylabel('Correlation with S&P 100', fontsize=12)
        ax.set_title('Portfolio Size vs Correlation', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for min and max correlation
        max_corr_idx = np.argmax(correlations)
        min_corr_idx = np.argmin(correlations)
        
        ax.annotate(f'Max: {correlations[max_corr_idx]:.4f}',
                    xy=(q_values[max_corr_idx], correlations[max_corr_idx]),
                    xytext=(10, 10), textcoords='offset points')
        
        ax.annotate(f'Min: {correlations[min_corr_idx]:.4f}',
                    xy=(q_values[min_corr_idx], correlations[min_corr_idx]),
                    xytext=(10, -10), textcoords='offset points')
        
        return fig, correlations, q_values

def main():
    st.set_page_config(page_title="S&P 100 Index Fund Optimization", layout="wide")
    
    st.title("S&P 100 Index Fund Optimization")
    st.markdown("""
    ### Project: Artificial Intelligence Driven Decision Making
    This application implements portfolio optimization to track the S&P 100 index using fewer stocks.
    """)
    
    # Parameters
    st.sidebar.header("Parameters")
    period = st.sidebar.selectbox("Analysis Period", ['3mo', '6mo', '9mo', '1y'])
    
    # Add tabs for different analyses
    tab1, tab2 = st.tabs(["Single Portfolio Analysis", "Q vs Correlation Analysis"])
    
    with tab1:
        q = st.slider("Number of Stocks (q)", 10, 100, 20)
        
        if st.button("Run Single Analysis"):
            try:
                index_fund = SP100IndexFund()
                index_fund.download_data(period=period)
                weights = index_fund.optimize_portfolio(q)
                metrics = index_fund.calculate_metrics(weights)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Selected Stocks and Weights")
                    weights_df = pd.DataFrame(list(weights.items()), columns=['Stock', 'Weight'])
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(weights_df)
                
                with col2:
                    st.subheader("Performance Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Correlation', 'Tracking Error (%)', 'Excess Return (%)'],
                        'Value': [
                            f"{metrics['correlation']:.4f}",
                            f"{metrics['tracking_error']:.2f}",
                            f"{metrics['excess_return']:.2f}"
                        ]
                    })
                    st.dataframe(metrics_df)
                
                # Plot performance
                st.subheader("Performance Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                (1 + metrics['portfolio_returns']).cumprod().plot(label='Optimized Portfolio', ax=ax)
                (1 + metrics['benchmark_returns']).cumprod().plot(label='S&P 100', ax=ax)
                plt.title('Cumulative Returns Comparison')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        if st.button("Run Q vs Correlation Analysis"):
            try:
                index_fund = SP100IndexFund()
                index_fund.download_data(period=period)
                
                # Run analysis for different q values
                fig, correlations, q_values = index_fund.analyze_q_vs_correlation(
                    q_range=range(10, 101, 10),
                    period=period
                )
                
                # Display the plot
                st.pyplot(fig)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                summary_df = pd.DataFrame({
                    'Metric': ['Maximum Correlation', 'Minimum Correlation', 'Average Correlation'],
                    'Value': [
                        f"{max(correlations):.4f} (q={q_values[np.argmax(correlations)]})",
                        f"{min(correlations):.4f} (q={q_values[np.argmin(correlations)]})",
                        f"{np.mean(correlations):.4f}"
                    ]
                })
                st.dataframe(summary_df)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 