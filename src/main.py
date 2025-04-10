from data_loader import SP100DataLoader
from optimization import PortfolioOptimizer
from ml_approach import MLPortfolioSelector
from performance import calculate_metrics
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Initialize components
    data_loader = SP100DataLoader()
    optimizer = PortfolioOptimizer()
    ml_selector = MLPortfolioSelector()
    
    # Parameters
    periods = ['3mo', '6mo', '9mo', '1y']
    q = 20  # Number of stocks to select
    
    # Store results
    results = {}
    
    for period in periods:
        print(f"\nAnalyzing {period} period...") 