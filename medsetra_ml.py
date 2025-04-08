import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Enable interactive mode for matplotlib
plt.ion()

# Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path)
   
    # Extract numerical columns for prediction
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
   
    # Create a new dataframe with just the numerical columns
    df_numeric = df[numerical_cols].copy()
   
    # Fill missing values
    df_numeric = df_numeric.fillna(df_numeric.mean())
   
    # Create a synthetic healthcare demand metric based on SDoH factors
    if 'ACS_PCT_DISABLE' in df_numeric.columns:
        df_numeric['HEALTHCARE_DEMAND'] = df_numeric['ACS_PCT_DISABLE'] * 0.3
    else:
        # Use other relevant columns if ACS_PCT_DISABLE not available
        demand_factors = [col for col in df_numeric.columns if 'PCT' in col and 'LT_HS' in col or 'MEDICAID' in col or 'UNINSURED' in col]
        if demand_factors:
            df_numeric['HEALTHCARE_DEMAND'] = df_numeric[demand_factors].mean(axis=1) * 2
        else:
            # If none of the specific columns exist, use mean of all columns
            df_numeric['HEALTHCARE_DEMAND'] = df_numeric.mean(axis=1)
   
    # Scale the healthcare demand to a reasonable range (0-100)
    min_max_scaler = MinMaxScaler(feature_range=(0, 100))
    df_numeric['HEALTHCARE_DEMAND'] = min_max_scaler.fit_transform(df_numeric['HEALTHCARE_DEMAND'].values.reshape(-1, 1))
   
    # For time series prediction, let's create dates - one day per record
    start_date = pd.Timestamp.now() - pd.Timedelta(days=len(df_numeric))
    dates = pd.date_range(start=start_date, periods=len(df_numeric), freq='D')
   
    # Create a new DataFrame with dates as index
    time_series_df = pd.DataFrame({'HEALTHCARE_DEMAND': df_numeric['HEALTHCARE_DEMAND'].values}, index=dates)
   
    return time_series_df, df_numeric

# Exploratory Data Analysis
def perform_eda(df, df_numeric):
    """Enhanced EDA function with additional visualizations"""
    # Create a figure for the time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['HEALTHCARE_DEMAND'])
    plt.title('Time Series of Healthcare Demand', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Healthcare Demand Index', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('healthcare_demand_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively
    
    # Distribution of healthcare demand
    plt.figure(figsize=(10, 6))
    sns.histplot(df['HEALTHCARE_DEMAND'], kde=True)
    plt.title('Distribution of Healthcare Demand', fontsize=16)
    plt.xlabel('Healthcare Demand Index', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('healthcare_demand_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively
    
    # Rolling statistics
    plt.figure(figsize=(12, 8))
    
    # Original series
    plt.subplot(311)
    plt.plot(df.index, df['HEALTHCARE_DEMAND'], label='Original')
    plt.title('Original Healthcare Demand', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Rolling mean
    plt.subplot(312)
    rolling_mean = df['HEALTHCARE_DEMAND'].rolling(window=7).mean()
    plt.plot(df.index, rolling_mean, label='Rolling Mean (7-day)', color='orange')
    plt.title('7-Day Rolling Mean', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Rolling std
    plt.subplot(313)
    rolling_std = df['HEALTHCARE_DEMAND'].rolling(window=7).std()
    plt.plot(df.index, rolling_std, label='Rolling Std (7-day)', color='green')
    plt.title('7-Day Rolling Standard Deviation', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('healthcare_demand_rolling_stats.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively
    
    # Correlation between SDoH factors and healthcare demand
    if len(df_numeric.columns) > 5:  # Only if we have enough columns
        # Select relevant SDoH columns
        sdoh_cols = [col for col in df_numeric.columns if col != 'HEALTHCARE_DEMAND'][:10]  # Limit to top 10 to avoid clutter
        if sdoh_cols:
            corr_matrix = df_numeric[sdoh_cols + ['HEALTHCARE_DEMAND']].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Between SDoH Factors and Healthcare Demand', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('healthcare_demand_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()  # Show the plot interactively
    
    return

# SARIMA Model Implementation
def implement_sarima(df, train_size):
    df = df.asfreq('D')
    train = df[:train_size]
    test = df[train_size:]
   
    try:
        model = SARIMAX(train['HEALTHCARE_DEMAND'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit(disp=False)
       
        forecast = results.get_forecast(steps=len(test))
        forecast_mean = forecast.predicted_mean
        rmse = np.sqrt(mean_squared_error(test['HEALTHCARE_DEMAND'], forecast_mean))
    except Exception as e:
        print(f"SARIMA error: {e}")
        # Fallback to simple moving average
        forecast_mean = train['HEALTHCARE_DEMAND'].rolling(window=7).mean().iloc[-1:].values[0]
        forecast_mean = np.array([forecast_mean] * len(test))
        rmse = np.sqrt(mean_squared_error(test['HEALTHCARE_DEMAND'], forecast_mean))
       
    return forecast_mean, rmse

# LSTM Model Implementation
def implement_lstm(df, train_size, look_back=7):
    values = df['HEALTHCARE_DEMAND'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
   
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
   
    # Handle case with limited data
    if len(scaled_values) <= look_back + 1:
        print("Not enough data for LSTM model")
        test_data = df[train_size:]
        lstm_forecast = np.array([values[-1][0]] * len(test_data))
        rmse = 0
        return lstm_forecast, rmse
   
    # Create sequence data
    X, y = create_sequences(scaled_values, look_back)
   
    train_samples = min(train_size - look_back, len(X) - 1)
    if train_samples <= 0:
        train_samples = max(1, int(len(X) * 0.8))
   
    # Split into train and test sets
    X_train, X_test = X[:train_samples], X[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
   
    if len(X_test) == 0:  # All data used for training
        X_test = X_train[-1:].copy()
        y_test = y_train[-1:].copy()
   
    try:
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(16))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
       
        # Compile and fit model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=min(16, len(X_train)), verbose=0, shuffle=False)
       
        # Make predictions
        test_predict = model.predict(X_test)
       
        # Invert scaling
        test_predict_reshaped = np.concatenate([test_predict], axis=1)
        test_predict_inv = scaler.inverse_transform(test_predict_reshaped)[:, 0]
       
        y_test_reshaped = np.concatenate([y_test], axis=1)
        y_test_inv = scaler.inverse_transform(y_test_reshaped)[:, 0]
       
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
       
        # Create a forecast aligned to test data timestamps
        test_data = df[train_size:]
       
        # Adjust the prediction length to match test data
        min_length = min(len(test_predict_inv), len(test_data))
        lstm_forecast = test_predict_inv[:min_length]
    except Exception as e:
        print(f"LSTM error: {e}")
        # Fallback
        test_data = df[train_size:]
        lstm_forecast = np.array([values[-1][0]] * len(test_data))
        rmse = 0
   
    return lstm_forecast, rmse

# Random Forest Model Implementation
def implement_random_forest(df, train_size, look_back=7):
    values = df['HEALTHCARE_DEMAND'].values
   
    def create_features(data, look_back=1):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)
   
    if len(values) <= look_back:
        print("Not enough data for Random Forest model")
        test_data = df[train_size:]
        rf_forecast = np.array([values[-1]] * len(test_data))
        rmse = 0
        return rf_forecast, rmse
   
    # Create sequence data
    X, y = create_features(values, look_back)
   
    train_samples = min(train_size - look_back, len(X) - 1)
    if train_samples <= 0:
        train_samples = max(1, int(len(X) * 0.8))
   
    # Split into train and test sets
    X_train, X_test = X[:train_samples], X[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
   
    if len(X_test) == 0:  # All data used for training
        X_test = X_train[-1:].copy()
        y_test = y_train[-1:].copy()
   
    try:
        # Build and train Random Forest model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
       
        # Make predictions
        test_predict = model.predict(X_test)
       
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, test_predict))
       
        # Create a forecast aligned to test data timestamps
        test_data = df[train_size:]
       
        # Adjust the prediction length to match test data
        min_length = min(len(test_predict), len(test_data))
        rf_forecast = test_predict[:min_length]
    except Exception as e:
        print(f"Random Forest error: {e}")
        # Fallback
        test_data = df[train_size:]
        rf_forecast = np.array([values[-1]] * len(test_data))
        rmse = 0
   
    return rf_forecast, rmse

# VAR Model Implementation
def implement_var(df, train_size, max_lags=7):
    df_copy = df.copy()
    df_copy = df_copy.asfreq('D')
   
    # Add lag features
    for i in range(1, 4):
        df_copy[f'HEALTHCARE_DEMAND_lag_{i}'] = df_copy['HEALTHCARE_DEMAND'].shift(i)
   
    # Add some additional features
    df_copy['HEALTHCARE_DEMAND_MA7'] = df_copy['HEALTHCARE_DEMAND'].rolling(window=7).mean()
    df_copy['HEALTHCARE_DEMAND_diff'] = df_copy['HEALTHCARE_DEMAND'].diff()
   
    # Drop NaN values
    df_copy = df_copy.dropna()
   
    if len(df_copy) <= max_lags:
        print("Not enough data for VAR model")
        test_data = df[train_size:]
        var_forecast = np.array([df['HEALTHCARE_DEMAND'].iloc[-1]] * len(test_data))
        rmse = 0
        return var_forecast, rmse
   
    # Split into train and test sets
    if len(df_copy) <= train_size:
        train_size = max(max_lags + 1, int(len(df_copy) * 0.8))
   
    train = df_copy[:train_size]
    test = df_copy[train_size:]
   
    # Ensure we have some test data
    if len(test) == 0:
        test = train[-1:].copy()
   
    try:
        # Fit VAR model
        model = VAR(train)
        results = model.fit(maxlags=min(max_lags, len(train) // 2 - 1))
       
        # Make predictions
        lag_order = results.k_ar
        forecast_input = train.values[-lag_order:]
        var_forecast = []
       
        for i in range(len(test)):
            forecast = results.forecast(y=forecast_input, steps=1)
            var_forecast.append(forecast[0][0])  # First column (HEALTHCARE_DEMAND) of the forecast
           
            # Update forecast input for next prediction
            forecast_input = np.vstack([forecast_input[1:], forecast])
           
            # Break if we've reached the end of test data
            if i >= len(test) - 1:
                break
       
        # Calculate RMSE
        var_forecast = np.array(var_forecast)
        min_length = min(len(var_forecast), len(test))
        var_forecast = var_forecast[:min_length]
       
        rmse = np.sqrt(mean_squared_error(test['HEALTHCARE_DEMAND'][:min_length], var_forecast))
    except Exception as e:
        print(f"VAR model error: {e}")
        # Fallback to simple moving average
        healthcare_demand = df['HEALTHCARE_DEMAND'].values
        train_demand = healthcare_demand[:train_size]
        window_size = min(7, len(train_demand))
        var_forecast = []
       
        for i in range(len(df) - train_size):
            if i < window_size:
                forecast = np.mean(train_demand[-window_size:])
            else:
                forecast = np.mean(var_forecast[-window_size:])
            var_forecast.append(forecast)
       
        var_forecast = np.array(var_forecast)
        test_data = df[train_size:]
        min_length = min(len(var_forecast), len(test_data))
       
        if min_length > 0:
            rmse = np.sqrt(mean_squared_error(test_data['HEALTHCARE_DEMAND'][:min_length], var_forecast[:min_length]))
        else:
            rmse = 0
        print(f"Using fallback moving average model. RMSE: {rmse:.4f}")
   
    return var_forecast, rmse

# XGBoost Model Implementation
def implement_xgboost(df, train_size, look_back=7):
    values = df['HEALTHCARE_DEMAND'].values
   
    def create_features(data, look_back=1):
        X, y = [], []
        for i in range(look_back, len(data)):
            features = list(data[i-look_back:i])
            X.append(features)
            y.append(data[i])
        return np.array(X), np.array(y)
   
    if len(values) <= look_back:
        print("Not enough data for XGBoost model")
        test_data = df[train_size:]
        xgb_forecast = np.array([values[-1]] * len(test_data))
        rmse = 0
        return xgb_forecast, rmse
   
    # Create sequence data
    X, y = create_features(values, look_back)
   
    train_samples = min(train_size - look_back, len(X) - 1)
    if train_samples <= 0:
        train_samples = max(1, int(len(X) * 0.8))
   
    # Split into train and test sets
    X_train, X_test = X[:train_samples], X[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
   
    if len(X_test) == 0:  # All data used for training
        X_test = X_train[-1:].copy()
        y_test = y_train[-1:].copy()
   
    try:
        # Build and train XGBoost model
        model = XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
       
        # Make predictions
        test_predict = model.predict(X_test)
       
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, test_predict))
       
        # Create a forecast aligned to test data timestamps
        test_data = df[train_size:]
       
        # Adjust the prediction length to match test data
        min_length = min(len(test_predict), len(test_data))
        xgb_forecast = test_predict[:min_length]
    except Exception as e:
        print(f"XGBoost error: {e}")
        # Fallback
        test_data = df[train_size:]
        xgb_forecast = np.array([values[-1]] * len(test_data))
        rmse = 0
   
    return xgb_forecast, rmse

# Hybrid Model Creation with five models
def create_hybrid_model(sarima_forecast, lstm_forecast, rf_forecast, var_forecast, xgb_forecast, test_data):
    if hasattr(sarima_forecast, 'predicted_mean'):
        sarima_values = sarima_forecast.predicted_mean
    else:
        sarima_values = sarima_forecast
   
    # Ensuring all forecasts have the same length
    min_length = min(len(sarima_values), len(lstm_forecast), len(rf_forecast), len(var_forecast), len(xgb_forecast))
   
    if min_length == 0:
        print("No valid forecast data available")
        return np.array([]), 0
   
    sarima_values = sarima_values[:min_length]
    lstm_values = lstm_forecast[:min_length]
    rf_values = rf_forecast[:min_length]
    var_values = var_forecast[:min_length]
    xgb_values = xgb_forecast[:min_length]
   
    # Combine forecasts with equal weights (1/5 each)
    hybrid_forecast = (1/5) * sarima_values + (1/5) * lstm_values + \
                     (1/5) * rf_values + (1/5) * var_values + (1/5) * xgb_values
   
    # Calculate RMSE
    if len(test_data) >= min_length:
        rmse = np.sqrt(mean_squared_error(test_data['HEALTHCARE_DEMAND'][:min_length], hybrid_forecast))
    else:
        rmse = 0
   
    return hybrid_forecast, rmse

# Function to plot model comparison
def plot_model_comparison(test_data, sarima_forecast, lstm_forecast, rf_forecast, var_forecast, xgb_forecast, hybrid_forecast):
    min_length = min(len(sarima_forecast), len(lstm_forecast), len(rf_forecast), 
                     len(var_forecast), len(xgb_forecast), len(hybrid_forecast), len(test_data))
    
    if min_length == 0:
        print("No data to plot for model comparison")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot actual values
    plt.plot(test_data.index[:min_length], test_data['HEALTHCARE_DEMAND'][:min_length], 
             label='Actual', linewidth=2, color='black')
    
    # Plot model predictions
    plt.plot(test_data.index[:min_length], sarima_forecast[:min_length], 
             label='SARIMA', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index[:min_length], lstm_forecast[:min_length], 
             label='LSTM', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index[:min_length], rf_forecast[:min_length], 
             label='Random Forest', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index[:min_length], var_forecast[:min_length], 
             label='VAR', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index[:min_length], xgb_forecast[:min_length], 
             label='XGBoost', linewidth=1.5, linestyle='--')
    
    # Plot hybrid forecast
    plt.plot(test_data.index[:min_length], hybrid_forecast[:min_length], 
             label='Hybrid Model', linewidth=2.5, color='red')
    
    plt.title('Comparison of Different Prediction Models', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Healthcare Demand Index', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively

# Threshold Setting and Alert System
def set_threshold_and_alert(hybrid_forecast, test_data):
    # Ensuring that we're using the same number of data points
    min_length = min(len(hybrid_forecast), len(test_data))
   
    if min_length == 0:
        print("No valid data for threshold calculation")
        return 0, []
   
    hybrid_forecast = hybrid_forecast[:min_length]
    test_data_subset = test_data.iloc[:min_length]
   
    # Set threshold as mean + 1.5 standard deviations (can be adjusted)
    threshold = np.mean(test_data_subset['HEALTHCARE_DEMAND']) + 1.5 * np.std(test_data_subset['HEALTHCARE_DEMAND'])
    alerts = []
   
    for i, demand in enumerate(hybrid_forecast):
        if demand > threshold:
            alert_date = test_data.index[i].strftime('%Y-%m-%d')
            alerts.append(f"Alert: Predicted healthcare demand {demand:.2f} exceeds threshold {threshold:.2f} on {alert_date}")
   
    return threshold, alerts

# Model Evaluation and Future Prediction
def evaluate_and_predict(df, hybrid_forecast, test_data, num_future_days=30):
    # Ensure that we're using the same number of data points
    min_length = min(len(hybrid_forecast), len(test_data))
   
    if min_length == 0:
        print("No valid data for evaluation")
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days)
        future_predictions = np.array([df['HEALTHCARE_DEMAND'].iloc[-1]] * num_future_days)
        return future_dates, future_predictions, {}
   
    hybrid_forecast = hybrid_forecast[:min_length]
    test_data_subset = test_data.iloc[:min_length]
   
    # Calculate evaluation metrics
    mse = mean_squared_error(test_data_subset['HEALTHCARE_DEMAND'], hybrid_forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data_subset['HEALTHCARE_DEMAND'], hybrid_forecast)
   
    # To avoid division by zero
    non_zero_actual = test_data_subset['HEALTHCARE_DEMAND'] != 0
    if non_zero_actual.any():
        mape = np.mean(np.abs((test_data_subset['HEALTHCARE_DEMAND'][non_zero_actual] - hybrid_forecast[non_zero_actual]) /
                              test_data_subset['HEALTHCARE_DEMAND'][non_zero_actual])) * 100
    else:
        mape = 0
   
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
   
    # Generate future dates
    last_date = test_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days)
   
    # For future predictions, use a simple approach based on the hybrid forecast trend
    # Could be improved with more sophisticated methods
    future_predictions = []
   
    # Use the last few values to determine a trend
    trend_window = min(14, len(hybrid_forecast))
    recent_values = hybrid_forecast[-trend_window:]
    trend = np.polyfit(range(trend_window), recent_values, 1)[0]
   
    # Use the last value as a starting point
    last_value = hybrid_forecast[-1]
   
    # Generate future predictions with trend and some randomness
    for i in range(num_future_days):
        next_value = last_value + trend * (i+1) + np.random.normal(0, np.std(recent_values) * 0.2)
        future_predictions.append(max(0, next_value))  # Ensure non-negative values
   
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return future_dates, np.array(future_predictions), metrics

# Function to plot results with enhanced visualizations
def plot_results(train_data, test_data, hybrid_forecast, future_dates, future_predictions, threshold, metrics):
    # Main forecast plot
    plt.figure(figsize=(14, 7))
   
    # Plot historical data
    plt.plot(train_data.index, train_data['HEALTHCARE_DEMAND'],
             label='Historical Data', color='blue', linewidth=1.5)
   
    # Plot test data (actual values)
    plt.plot(test_data.index, test_data['HEALTHCARE_DEMAND'],
             label='Actual Demand', color='green', linewidth=1.5)
   
    # Plot hybrid forecast (predicted values)
    min_length = min(len(hybrid_forecast), len(test_data))
    plt.plot(test_data.index[:min_length], hybrid_forecast[:min_length],
             label='Predicted Demand', color='orange', linewidth=2, linestyle='--')
   
    # Plot future predictions
    plt.plot(future_dates, future_predictions,
             label='Future Demand Forecast', color='red', linewidth=2, linestyle='-.')
   
    # Plot threshold line
    plt.axhline(y=threshold, color='purple', linestyle='--',
                label=f'Demand Spike Threshold ({threshold:.2f})', linewidth=1.5)
   
    # Highlight areas above threshold in the future
    above_threshold = future_predictions > threshold
    if any(above_threshold):
        plt.fill_between(future_dates, 0, future_predictions,
                         where=above_threshold, color='red', alpha=0.3,
                         label='Predicted Demand Spikes')
   
    plt.title('Healthcare Demand Forecast Based on SDoH Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Healthcare Demand Index', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
   
    plt.savefig('healthcare_demand_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively

    # Additional plot: Focus on future predictions
    plt.figure(figsize=(14, 7))
    plt.plot(future_dates, future_predictions,
             label='Future Demand Forecast', color='blue', linewidth=2)
    plt.axhline(y=threshold, color='red', linestyle='--',
                label=f'Demand Spike Threshold ({threshold:.2f})', linewidth=1.5)
   
    # Highlight areas above threshold
    plt.fill_between(future_dates, 0, future_predictions,
                     where=above_threshold, color='red', alpha=0.3,
                     label='Predicted Demand Spikes')
   
    plt.title('30-Day Healthcare Demand Forecast', fontsize=16)
    # Continuation of plot_results function
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Healthcare Demand Index', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('future_healthcare_demand_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively
    
    # Create a metrics summary plot
    plt.figure(figsize=(10, 6))
    metrics_values = [metrics['rmse'], metrics['mae'], metrics['mape']]
    metrics_labels = ['RMSE', 'MAE', 'MAPE (%)']
    
    bars = plt.bar(metrics_labels, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Model Performance Metrics', fontsize=16)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show the plot interactively

# Generate a report for healthcare providers
def generate_report(df, future_dates, future_predictions, threshold, alerts, metrics):
    report = """
# Healthcare Demand Forecast Report

## Summary
This report presents a forecast of healthcare demand based on Social Determinants of Health (SDoH) data.

## Key Findings
- **Average Projected Demand:** {:.2f}
- **Maximum Projected Demand:** {:.2f} (on {})
- **Forecasted Trend:** {}
- **Number of Alert Days:** {}

## Model Performance
- **RMSE:** {:.4f}
- **MAE:** {:.4f}
- **MAPE:** {:.4f}%

## Alerts for Next 30 Days
{}

## Recommendations
{}
    """.format(
        np.mean(future_predictions),
        np.max(future_predictions),
        future_dates[np.argmax(future_predictions)].strftime('%Y-%m-%d'),
        "Increasing" if future_predictions[-1] > future_predictions[0] else "Decreasing",
        len(alerts),
        metrics['rmse'],
        metrics['mae'],
        metrics['mape'],
        "\n".join(alerts) if alerts else "No alerts for the forecast period.",
        "- Prepare for potential demand spikes on dates listed in alerts.\n- Allocate resources accordingly to meet projected demand.\n- Consider preventive interventions in areas with high SDoH risk factors." if alerts else "- Maintain regular staffing and resource levels for the forecast period.\n- Continue monitoring for changes in SDoH factors."
    )
    
    # Save the report to a file
    with open('healthcare_demand_forecast_report.md', 'w') as f:
        f.write(report)
    
    print("Report generated and saved as 'healthcare_demand_forecast_report.md'")
    return report

# Main execution
def main(file_path, forecast_days=30):
    # Load and preprocess data
    df, df_numeric = load_and_preprocess_data(file_path)
    
    # Perform EDA
    perform_eda(df, df_numeric)
    
    # Determine train/test split
    train_size = int(len(df) * 0.8)  # 80% for training
    print(f"Training data size: {train_size}, Test data size: {len(df) - train_size}")
    
    # Split data
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # Implement models
    print("Implementing SARIMA model...")
    sarima_forecast, sarima_rmse = implement_sarima(df, train_size)
    print(f"SARIMA RMSE: {sarima_rmse:.4f}")
    
    print("Implementing LSTM model...")
    lstm_forecast, lstm_rmse = implement_lstm(df, train_size)
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    
    print("Implementing Random Forest model...")
    rf_forecast, rf_rmse = implement_random_forest(df, train_size)
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    
    print("Implementing VAR model...")
    var_forecast, var_rmse = implement_var(df, train_size)
    print(f"VAR RMSE: {var_rmse:.4f}")
    
    print("Implementing XGBoost model...")
    xgb_forecast, xgb_rmse = implement_xgboost(df, train_size)
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    
    # Create hybrid model
    print("Creating hybrid model...")
    hybrid_forecast, hybrid_rmse = create_hybrid_model(
        sarima_forecast, lstm_forecast, rf_forecast, var_forecast, xgb_forecast, test_data)
    print(f"Hybrid Model RMSE: {hybrid_rmse:.4f}")
    
    # Plot model comparison
    plot_model_comparison(test_data, sarima_forecast, lstm_forecast, rf_forecast, 
                         var_forecast, xgb_forecast, hybrid_forecast)
    
    # Set threshold and generate alerts
    threshold, alerts = set_threshold_and_alert(hybrid_forecast, test_data)
    print(f"Demand spike threshold set at: {threshold:.2f}")
    if alerts:
        print("Alerts generated:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("No alerts generated for the test period.")
    
    # Predict future values
    future_dates, future_predictions, metrics = evaluate_and_predict(df, hybrid_forecast, test_data, forecast_days)
    
    # Plot results
    plot_results(train_data, test_data, hybrid_forecast, future_dates, future_predictions, threshold, metrics)
    
    # Generate report
    report = generate_report(df, future_dates, future_predictions, threshold, alerts, metrics)
    print("Analysis complete. Check the generated images and report for detailed results.")
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'hybrid_forecast': hybrid_forecast,
        'future_dates': future_dates,
        'future_predictions': future_predictions,
        'threshold': threshold,
        'alerts': alerts,
        'metrics': metrics,
        'report': report
    }

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "healthcare_sdoh_data.csv"
    
    # Check if file exists
    import os
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please provide a valid path to a CSV file containing healthcare and SDoH data.")
        print("Creating example data for demonstration...")
        
        # Create example data if file not found
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        np.random.seed(42)
        
        # Create synthetic data
        base = 50 + np.random.normal(0, 5, len(dates))
        trend = np.linspace(0, 15, len(dates))
        seasonal = 10 * np.sin(np.linspace(0, 2*np.pi*4, len(dates)))
        
        demand = base + trend + seasonal
        demand = np.maximum(0, demand)  # Ensure non-negative values
        
        # Create DataFrame
        example_df = pd.DataFrame({
            'Date': dates,
            'ACS_PCT_DISABLE': np.random.uniform(5, 15, len(dates)),
            'ACS_PCT_UNINSURED': np.random.uniform(8, 20, len(dates)),
            'ACS_PCT_LT_HS': np.random.uniform(10, 25, len(dates)),
            'ACS_PCT_MEDICAID': np.random.uniform(15, 30, len(dates)),
            'ACS_INCOME_RATIO': np.random.uniform(0.5, 3, len(dates)),
            'HEALTHCARE_DEMAND': demand
        })
        
        # Save example data
        example_file_path = "example_healthcare_sdoh_data.csv"
        example_df.to_csv(example_file_path, index=False)
        
        print(f"Example data created and saved to '{example_file_path}'")
        file_path = example_file_path
    
    # Run the main analysis
    results = main(file_path, forecast_days=30)
    
    print("End of analysis. Thank you for using the Healthcare Demand Forecasting tool.")
