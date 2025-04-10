import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SP100IndexFund:
    def __init__(self):
        self.data = None
        self.benchmark = None

    def download_data(self, period='3mo'):
        """Download S&P 100 data"""
        with st.spinner('Downloading S&P 100 data...'):
            try:
                sp100 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[2]
                tickers = sp100['Symbol'].tolist()
                
                data = {}
                progress_bar = st.progress(0)
                for i, ticker in enumerate(tickers):
                    try:
                        if ticker == 'BRK.B':
                            ticker = 'BRK-B'
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period=period)
                        if not hist.empty:
                            data[ticker] = hist['Close']
                    except Exception as e:
                        st.warning(f"Error downloading {ticker}: {str(e)}")
                    finally:
                        progress_bar.progress((i + 1) / len(tickers))
                
                self.data = pd.DataFrame(data)
                
                # Download benchmark
                benchmark = yf.Ticker('^OEX')
                self.benchmark = benchmark.history(period=period)['Close']
                
                return self.data
            except Exception as e:
                st.error(f"Error: {str(e)}")
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

    def optimize_correlation(self, q=20):
        """Simple correlation-based approach for comparison"""
        if self.data is None or self.benchmark is None:
            raise ValueError("Data not downloaded")

        returns = self.data.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()

        # Calculate correlations
        correlations = {}
        for column in returns.columns:
            correlation = returns[column].corr(benchmark_returns)
            if not np.isnan(correlation):
                correlations[column] = abs(correlation)

        # Select top q stocks
        sorted_stocks = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:q]
        
        # Normalize weights
        total_correlation = sum(corr for _, corr in sorted_stocks)
        weights = {stock: corr/total_correlation for stock, corr in sorted_stocks}
        
        return weights

def main():
    st.set_page_config(page_title="S&P 100 Index Fund Optimization", layout="wide")
    
    st.title("S&P 100 Index Fund Optimization")
    st.markdown("""
    ### Project: Artificial Intelligence Driven Decision Making
    This application implements two approaches for index fund optimization:
    1. **Mathematical Optimization**: Using scipy's SLSQP optimizer
    2. **Correlation-Based**: Simple statistical approach for comparison
    """)
    
    # Parameters
    st.sidebar.header("Parameters")
    q = st.sidebar.slider("Number of Stocks (q)", 10, 100, 20)
    period = st.sidebar.selectbox("Analysis Period", ['3mo', '6mo', '9mo', '1y'])
    
    if st.button("Run Analysis"):
        try:
            index_fund = SP100IndexFund()
            
            # Download data
            index_fund.download_data(period=period)
            
            # Run both methods
            with st.spinner("Running mathematical optimization..."):
                opt_weights = index_fund.optimize_portfolio(q)
            
            with st.spinner("Running correlation-based approach..."):
                corr_weights = index_fund.optimize_correlation(q)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimization Results")
                opt_df = pd.DataFrame(opt_weights.items(), 
                                    columns=['Stock', 'Weight'])
                opt_df['Weight'] = opt_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(opt_df)
            
            with col2:
                st.subheader("Correlation-Based Results")
                corr_df = pd.DataFrame(corr_weights.items(), 
                                     columns=['Stock', 'Weight'])
                corr_df['Weight'] = corr_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(corr_df)
            
            # Performance comparison
            st.subheader("Performance Comparison")
            returns = index_fund.data.pct_change()
            benchmark_returns = index_fund.benchmark.pct_change()
            
            # Calculate portfolio returns
            opt_returns = sum(returns[stock] * weight 
                            for stock, weight in opt_weights.items())
            corr_returns = sum(returns[stock] * weight 
                             for stock, weight in corr_weights.items())
            
            # Calculate metrics
            metrics = pd.DataFrame({
                'Optimization': {
                    'Correlation': opt_returns.corr(benchmark_returns),
                    'Tracking Error': np.std(opt_returns - benchmark_returns)
                },
                'Correlation-Based': {
                    'Correlation': corr_returns.corr(benchmark_returns),
                    'Tracking Error': np.std(corr_returns - benchmark_returns)
                }
            })
            
            st.dataframe(metrics)
            
            # Plot cumulative returns
            fig, ax = plt.subplots(figsize=(12, 6))
            (1 + opt_returns).cumprod().plot(label='Optimized Portfolio', ax=ax)
            (1 + corr_returns).cumprod().plot(label='Correlation Portfolio', ax=ax)
            (1 + benchmark_returns).cumprod().plot(label='S&P 100', ax=ax)
            plt.title('Cumulative Returns Comparison')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 