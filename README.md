# S&P 100 Index Fund Optimizer

## Project Description
This project implements an index fund optimization tool that tracks the S&P 100 using a smaller subset of stocks. It's part of the Artificial Intelligence Driven Decision Making (MSCAI1) course project.

## Features
- Mathematical optimization using scipy's SLSQP optimizer
- Correlation-based stock selection for comparison
- Performance analysis across multiple time periods
- Interactive visualization of results

## Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure
```
fundforge/
├── app/                 # Next.js application
├── components/          # React components
├── contracts/           # Smart contracts
├── public/             # Static assets
├── styles/             # Global styles
└── utils/              # Utility functions
```

## License

MIT