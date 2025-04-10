import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def get_sp100_tickers():
    """Get the list of S&P 100 tickers."""
    # S&P 100 components as of 2024
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK.B', 'JPM', 'JNJ', 'V',
        'WMT', 'MA', 'PG', 'NVDA', 'HD', 'BAC', 'PFE', 'KO', 'DIS', 'NFLX',
        'CMCSA', 'PEP', 'ABBV', 'TMO', 'CSCO', 'INTC', 'VZ', 'MRK', 'ABT', 'CVX',
        'MCD', 'WFC', 'ORCL', 'NKE', 'PM', 'T', 'UNH', 'ACN', 'IBM', 'LOW',
        'UPS', 'MS', 'BMY', 'AMGN', 'HON', 'COP', 'CAT', 'GS', 'BA', 'MMM',
        'RTX', 'UNP', 'PLD', 'LMT', 'ADP', 'ELV', 'BLK', 'AMT', 'TXN', 'DE',
        'SCHW', 'MDT', 'GILD', 'CI', 'AXP', 'ISRG', 'VRTX', 'LRCX', 'PGR', 'ADI',
        'REGN', 'NOW', 'PANW', 'MU', 'ZTS', 'SNPS', 'CDNS', 'KLAC', 'MCHP', 'ROP',
        'ADSK', 'ANET', 'FTNT', 'MRNA', 'KDP', 'CPRT', 'MELI', 'TEAM', 'CTAS', 'DXCM',
        'EXC', 'AEP', 'SRE', 'WEC', 'ED', 'DUK', 'SO', 'D', 'NEE', 'PCG'
    ]
    return sp100_tickers

def download_stock_data(tickers, start_date, end_date):
    """Download historical stock data for given tickers."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                data[ticker] = df['Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return pd.DataFrame(data)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Get S&P 100 tickers
    tickers = get_sp100_tickers()
    
    # Set date range for historical data (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Download data
    print("Downloading S&P 100 stock data...")
    df = download_stock_data(tickers, start_date, end_date)
    
    # Save to CSV
    output_file = '../data/sp100_data.csv'
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    # Calculate and save returns
    returns = df.pct_change().dropna()
    returns_file = '../data/sp100_returns.csv'
    returns.to_csv(returns_file)
    print(f"Returns data saved to {returns_file}")

if __name__ == "__main__":
    main() 