import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class SP100DataLoader:
    def __init__(self):
        self.sp100_ticker = '^OEX'
        self.data = None
        self.benchmark = None
        
    def get_sp100_components(self) -> list:
        """Get current S&P 100 components"""
        sp100 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[2]
        return sp100['Symbol'].tolist()
        
    def download_data(self, period: str = '1y') -> Tuple[pd.DataFrame, pd.Series]:
        """Download historical data for S&P 100 stocks and index
        
        Args:
            period: Data period ('3mo', '6mo', '9mo', '1y')
        """
        # Get S&P 100 components
        tickers = self.get_sp100_components()
        
        # Download stock data
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data[ticker] = stock.history(period=period)['Close']
            except:
                continue
                
        self.data = pd.DataFrame(data)
        
        # Download benchmark data
        benchmark = yf.Ticker(self.sp100_ticker)
        self.benchmark = benchmark.history(period=period)['Close']
        
        return self.data, self.benchmark 