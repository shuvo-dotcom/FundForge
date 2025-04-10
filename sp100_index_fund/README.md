# S&P 100 Index Fund Implementation

This project implements multiple approaches to create an index fund that tracks the S&P 100 using a subset of stocks. The goal is to maintain performance similar to the full S&P 100 index while using fewer stocks.

## Project Structure

```
sp100_index_fund/
├── data/               # Data storage
├── src/               # Source code
│   └── sp100_index.py # Main implementation
├── results/           # Output results and visualizations
└── requirements.txt   # Project dependencies
```

## Implementation Approaches

1. **Correlation-Based Optimization**
   - Uses CVXPY to solve a binary optimization problem
   - Maximizes correlation with the S&P 100 benchmark
   - Selects optimal subset of stocks and their weights

2. **Sector-Based Selection**
   - Maintains sector diversification
   - Selects top-performing stocks from each sector
   - Uses equal weighting within sectors

3. **Momentum-Based Selection**
   - Uses technical analysis indicators
   - Combines 3-month and 6-month momentum
   - Considers volatility-adjusted performance

4. **Machine Learning Approach**
   - Uses Random Forest regression
   - Features include technical indicators and volatility metrics
   - Predicts correlation with benchmark

## Performance Metrics

Each approach is evaluated using:
- Correlation with S&P 100 benchmark
- Tracking error
- Portfolio returns
- Benchmark returns

Results are analyzed across different time periods (3 months to 1 year).

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the implementation:
   ```bash
   python src/sp100_index.py
   ```

## Results

The script will:
1. Download historical data for S&P 100 stocks
2. Implement all four selection methods
3. Evaluate performance for different portfolio sizes (q = 20, 30, 40, 50)
4. Generate performance visualizations
5. Save results in the results/ directory

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Note

This implementation focuses solely on the S&P 100 index fund tracking problem and does not include any crowdfunding or campaign-related functionality. 