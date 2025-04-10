import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

class AIOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': self._build_neural_network(),
            'lstm': self._build_lstm_model()
        }
    
    def _build_neural_network(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(None,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_features(self, returns, benchmark_returns, lookback=20):
        features = pd.DataFrame()
        
        for ticker in returns.columns:
            stock_returns = returns[ticker]
            
            # Technical indicators
            rolling_mean = stock_returns.rolling(window=lookback).mean()
            rolling_std = stock_returns.rolling(window=lookback).std()
            rolling_skew = stock_returns.rolling(window=lookback).skew()
            rolling_kurt = stock_returns.rolling(window=lookback).kurt()
            
            # Momentum indicators
            momentum = stock_returns.rolling(window=lookback).sum()
            rsi = self._calculate_rsi(stock_returns, window=lookback)
            
            # Volatility indicators
            volatility = stock_returns.rolling(window=lookback).std()
            volatility_ratio = volatility / benchmark_returns.rolling(window=lookback).std()
            
            # Correlation features
            correlation = stock_returns.rolling(window=lookback).corr(benchmark_returns)
            
            # Combine features
            features.loc[ticker, 'mean_return'] = rolling_mean.iloc[-1]
            features.loc[ticker, 'volatility'] = rolling_std.iloc[-1]
            features.loc[ticker, 'skewness'] = rolling_skew.iloc[-1]
            features.loc[ticker, 'kurtosis'] = rolling_kurt.iloc[-1]
            features.loc[ticker, 'momentum'] = momentum.iloc[-1]
            features.loc[ticker, 'rsi'] = rsi.iloc[-1]
            features.loc[ticker, 'volatility_ratio'] = volatility_ratio.iloc[-1]
            features.loc[ticker, 'correlation'] = correlation.iloc[-1]
        
        # Handle missing values
        features = features.fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)
        
        return pd.DataFrame(pca_features, index=features.index)
    
    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_models(self, features, target):
        trained_models = {}
        
        # Prepare data for LSTM
        lstm_data = features.values.reshape((features.shape[0], features.shape[1], 1))
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.fit(lstm_data, target, epochs=50, batch_size=32, verbose=0)
            elif name == 'neural_network':
                model.fit(features, target, epochs=50, batch_size=32, verbose=0)
            else:
                model.fit(features, target)
            
            trained_models[name] = model
        
        return trained_models
    
    def predict_returns(self, features, model_name='ensemble'):
        if model_name == 'ensemble':
            predictions = []
            for name, model in self.models.items():
                if name == 'lstm':
                    pred = model.predict(features.values.reshape((features.shape[0], features.shape[1], 1)))
                else:
                    pred = model.predict(features)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        else:
            model = self.models[model_name]
            if model_name == 'lstm':
                return model.predict(features.values.reshape((features.shape[0], features.shape[1], 1)))
            return model.predict(features)
    
    def optimize_portfolio(self, returns, benchmark_returns, q=20, model_name='ensemble'):
        # Prepare features
        features = self.prepare_features(returns, benchmark_returns)
        
        # Train models
        trained_models = self.train_models(features, benchmark_returns)
        
        # Predict returns
        predicted_returns = self.predict_returns(features, model_name)
        
        # Select top q stocks
        selected_indices = np.argsort(predicted_returns)[-q:]
        selected_stocks = returns.columns[selected_indices]
        
        # Calculate covariance matrix for selected stocks
        cov_matrix = returns[selected_stocks].cov()
        
        # Optimize weights using mean-variance optimization
        n = len(selected_stocks)
        weights = np.ones(n) / n  # Equal weights as initial guess
        
        def objective(weights):
            portfolio_returns = returns[selected_stocks].dot(weights)
            tracking_error = np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))
            return tracking_error
        
        from scipy.optimize import minimize
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # Weights non-negative
        ]
        
        result = minimize(objective, weights, constraints=constraints, method='SLSQP')
        
        # Create weights dictionary
        weights_dict = {stock: weight for stock, weight in zip(selected_stocks, result.x)}
        
        return weights_dict, selected_stocks 