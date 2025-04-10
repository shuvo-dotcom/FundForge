from amplpy import AMPL, Environment
import pandas as pd
import numpy as np
from typing import Dict

class PortfolioOptimizer:
    def __init__(self):
        self.ampl = AMPL(Environment())
        
    def optimize_weights(self, returns: pd.DataFrame, q: int) -> Dict[str, float]:
        """Optimize portfolio weights using AMPL
        
        Args:
            returns: Stock returns data
            q: Number of stocks to select
            
        Returns:
            Dictionary of selected stocks and their weights
        """
        # Prepare data for AMPL
        T = returns.shape[0]  # Number of time periods
        N = returns.shape[1]  # Number of stocks
        
        # Load AMPL model
        self.ampl.read('models/portfolio.mod')
        
        # Set parameters
        self.ampl.param['T'] = T
        self.ampl.param['N'] = N
        self.ampl.param['q'] = q
        
        # Set data
        self.ampl.set['TIME'] = range(1, T+1)
        self.ampl.set['ASSETS'] = returns.columns.tolist()
        
        # Set returns data
        returns 