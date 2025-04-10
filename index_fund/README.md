# AI-Driven S&P 100 Index Fund

This project implements an AI-driven approach to create an index fund that tracks the S&P 100 using a subset of stocks. The goal is to maintain performance similar to the full S&P 100 index while using fewer stocks.

## Project Structure

```
index_fund/
├── data/               # Data storage
├── src/               # Source code
│   ├── data_collection.py    # Data collection and preprocessing
│   └── optimization.py       # Portfolio optimization
├── results/           # Output results and visualizations
└── requirements.txt   # Project dependencies
```

## Implementation Approaches

1. **Correlation-based Optimization (Primary Method)**
   - Uses CVXPY to solve a binary optimization problem
   - Maximizes correlation with the S&P 100 benchmark
   - Selects optimal subset of stocks and their weights

2. **Variance-based Optimization (Alternative Method)**
   - Minimizes tracking error variance
   - Uses the same optimization framework
   - Provides comparison with correlation-based approach

## Key Features

- Automated data collection using yfinance
- Flexible parameter selection for number of stocks (q)
- Performance evaluation across multiple time periods
- Visualization of results and metrics
- Support for different optimization methods

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Collect data:
   ```bash
   python src/data_collection.py
   ```

3. Run optimization:
   ```bash
   python src/optimization.py
   ```

## Performance Metrics

The system evaluates performance using:
- Correlation with benchmark
- Tracking error
- Portfolio returns
- Benchmark returns

Results are analyzed across different time periods (3 months to 1 year) to ensure robustness.

## Results

Performance metrics and visualizations are saved in the `results/` directory, including:
- Selected stocks and their weights
- Correlation and tracking error metrics
- Return comparisons
- Performance plots

## Team

This project is developed by a team of 2 participants as part of an AI-driven financial analysis project. 