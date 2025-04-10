# Portfolio Optimization Model for S&P 100 Index Fund

# Sets
set STOCKS;  # Set of selected stocks
set TIME;    # Set of time periods

# Parameters
param returns{STOCKS, TIME};  # Stock returns
param benchmark{TIME};        # Benchmark returns
param q;                      # Number of stocks to select
param min_weight;             # Minimum weight per stock
param max_weight;             # Maximum weight per stock

# Variables
var weights{STOCKS} >= min_weight, <= max_weight;  # Portfolio weights
var tracking_error;                                 # Tracking error

# Objective
minimize obj: tracking_error;

# Constraints
subject to weight_sum:
    sum{s in STOCKS} weights[s] = 1;

subject to tracking_error_def:
    tracking_error = sqrt(sum{t in TIME} 
        (sum{s in STOCKS} weights[s] * returns[s,t] - benchmark[t])^2 / card(TIME));

# Additional constraints can be added here
# For example:
# - Maximum sector exposure
# - Turnover constraints
# - Transaction cost constraints 