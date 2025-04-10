from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from typing import Dict

class MLPortfolioSelector:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def select_stocks(self, returns: pd.DataFrame, benchmark_returns: pd.Series, q: int) -> Dict[str, float]:
        """Select stocks using Random Forest feature importance
        
        Args:
            returns: Stock returns data
            benchmark_returns: Benchmark returns
            q: Number of stocks to select
            
        Returns:
            Dictionary of selected stocks and their weights
        """
        # Train Random Forest
        self.model.fit(returns, benchmark_returns)
        
        # Get feature importances
        importances = dict(zip(returns.columns, self.model.feature_importances_))
        
        # Select top q stocks
        selected = dict(sorted(importances.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:q])
                             
        # Calculate weights based on importance scores
        total_importance = sum(selected.values())
        weights = {k: v/total_importance for k, v in selected.items()}
        
        return weights 