# Sets
set TIME;    # Set of time periods
set ASSETS;  # Set of available assets

# Parameters
param T;     # Number of time periods
param N;     # Number of assets
param q;     # Number of assets to select
param returns{TIME, ASSETS};  # Asset returns

# Variables
var w{ASSETS} >= 0;          # Portfolio weights
var z{ASSETS} binary;        # Asset selection variables

# Objective: Maximize correlation with benchmark
maximize correlation:
    sum{t in TIME} (
        sum{i in ASSETS} w[i] * returns[t,i]
    );

# Constraints
subject to sum_weights:
    sum{i in ASSETS} w[i] = 1;

subject to num_assets:
    sum{i in ASSETS} z[i] = q;

subject to weight_limits{i in ASSETS}:
    w[i] <= z[i]; 