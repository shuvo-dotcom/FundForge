import pandas as pd
import numpy as np
from typing import Dict, Tuple

def calculate_metrics(returns: pd.DataFrame, 
                     weights: Dict[str, float], 
                     benchmark_returns: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics
    
    Args:
        returns: Stock returns data
        weights: Portfolio weights
        benchmark_returns: Benchmark returns
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate portfolio returns
    portfolio_returns = sum(returns[stock] * weight 
                          for stock, weight in weights.items())
    
    # Calculate metrics
    correlation = portfolio_returns.corr(benchmark_returns)
    tracking_error = np.std(portfolio_returns - benchmark_returns)
    information_ratio = (portfolio_returns.mean() - benchmark_returns.mean()) / tracking_error
    
    return {
        'correlation': correlation,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    } 