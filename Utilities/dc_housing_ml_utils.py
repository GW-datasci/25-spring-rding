"""
DC Housing Market Machine Learning Utilities
-------------------------------------------
This module contains utility functions for machine learning models
used in DC housing market price prediction.

Functions include data preprocessing, model building, evaluation metrics,
and visualization utilities for various models including:
- Lasso/ElasticNet regression
- Support Vector Regression (SVR)
- LSTM neural networks
- Historical mean benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout # Added Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
from scipy.stats import uniform, loguniform

# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# Common Evaluation Metrics
# =============================================================================

def mean_absolute_percentage_error_np(y_true, y_pred):
    """Numpy implementation of MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0: 
        return np.nan
    if len(y_true[mask]) == 0: 
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy_np(y_true, y_pred, y_true_prev):
    """Numpy implementation of directional accuracy."""
    y_true, y_pred, y_true_prev = np.array(y_true), np.array(y_pred), np.array(y_true_prev)
    min_len = min(len(y_true), len(y_pred), len(y_true_prev))
    if min_len == 0: 
        return np.nan
    
    y_true, y_pred, y_true_prev = y_true[:min_len], y_pred[:min_len], y_true_prev[:min_len]
    actual_diff = np.sign(y_true - y_true_prev)
    pred_diff = np.sign(y_pred - y_true_prev)
    correct_direction = (actual_diff == pred_diff).astype(int)
    return np.mean(correct_direction) * 100

def calculate_metrics(actuals, predictions, prev_actuals=None):
    """Calculate common regression metrics."""
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error_np(actuals, predictions)
    
    # Calculate directional accuracy if previous actuals are provided
    da = np.nan
    if prev_actuals is not None:
        da = directional_accuracy_np(actuals, predictions, prev_actuals)
        
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Dir_Acc': da
    }

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(file_path, date_col='Date', target_pct_change_col='House_Index_pct_change', 
              target_level_col='House_Index', final_date_str=None):
    """
    Load and prepare data for modeling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    date_col : str
        Name of the date column
    target_pct_change_col : str
        Name of the target percentage change column
    target_level_col : str
        Name of the target level column
    final_date_str : str, optional
        Date string to filter data up to
        
    Returns:
    --------
    df : DataFrame
        Processed DataFrame
    """
    try:
        # Load data
        df = pd.read_csv(file_path, parse_dates=[date_col])
        # Remove any unnamed columns and sort
        if date_col in df.columns:
            df = df.set_index(date_col)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.sort_index()
        print(f"Data loaded successfully from '{file_path}'.")
        
        # Filter by date if specified
        if final_date_str is not None:
            final_date = pd.to_datetime(final_date_str)
            df = df[df.index <= final_date].copy()
            print(f"Data filtered up to {final_date_str}. Shape: {df.shape}")
            
        # Check for required columns
        required_cols = [target_pct_change_col, target_level_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Handle missing values
        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        rows_after_na_drop = len(df)
        if initial_rows > rows_after_na_drop:
            print(f"Dropped {initial_rows - rows_after_na_drop} rows containing NaNs in target columns.")
            
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def split_data(df, train_end_date_str, test_start_date_str, feature_cols, 
               target_pct_change_col='House_Index_pct_change', 
               target_level_col='House_Index'):
    """
    Split data into training and test sets based on date.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with datetime index
    train_end_date_str : str
        End date for training set
    test_start_date_str : str
        Start date for test set
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target percentage change column
    target_level_col : str
        Name of the target level column
        
    Returns:
    --------
    X_train, y_train_pct_change, y_train_level, X_test, y_test_pct_change, y_test_level
    """
    train_end_date = pd.to_datetime(train_end_date_str)
    test_start_date = pd.to_datetime(test_start_date_str)
    
    X_train = df[df.index <= train_end_date][feature_cols]
    y_train_pct_change = df[df.index <= train_end_date][target_pct_change_col]
    y_train_level = df[df.index <= train_end_date][target_level_col]
    
    X_test = df[df.index >= test_start_date][feature_cols]
    y_test_pct_change = df[df.index >= test_start_date][target_pct_change_col]
    y_test_level = df[df.index >= test_start_date][target_level_col]
    
    if X_train.empty or X_test.empty:
        raise ValueError("Training or test set is empty after split")
        
    print(f"Training data: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
    
    return X_train, y_train_pct_change, y_train_level, X_test, y_test_pct_change, y_test_level

def identify_lagged_features(feature_cols, target_pct_change_col='House_Index_pct_change'):
    """
    Identify lagged features and extract lag numbers.
    
    Parameters:
    -----------
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target column
        
    Returns:
    --------
    house_index_features : list
        List of lagged target features
    lag_numbers : dict
        Dictionary mapping feature names to lag numbers
    exogenous_features : list
        List of exogenous features
    """
    # Identify target-related lagged features
    house_index_features = [col for col in feature_cols 
                            if target_pct_change_col in col and '_lag' in col]
    
    # Identify exogenous features
    exogenous_features = [col for col in feature_cols 
                          if col not in house_index_features]
    
    # Extract lag numbers from target features
    lag_numbers = {}
    for feature in house_index_features:
        parts = feature.split('_lag')
        if len(parts) > 1:
            try:
                lag_numbers[feature] = int(parts[1])
            except ValueError:
                print(f"Warning: Could not extract lag number from {feature}")
                continue
                
    print(f"Identified {len(house_index_features)} lagged target features")
    print(f"Identified {len(exogenous_features)} exogenous features")
    
    return house_index_features, lag_numbers, exogenous_features

# =============================================================================
# ElasticNet/Lasso Model Functions
# =============================================================================

def tune_linear_models(X_train_scaled, y_train_pct_change, n_cv_splits=9, 
                       cv_lasso_alphas=None, cv_elasticnet_alphas=None, 
                       cv_l1_ratios=None):
    """
    Tune Lasso and ElasticNet models using cross-validation.
    
    Parameters:
    -----------
    X_train_scaled : ndarray
        Scaled training features
    y_train_pct_change : Series
        Target percentage change values
    n_cv_splits : int
        Number of cross-validation splits
    cv_lasso_alphas : ndarray, optional
        Alpha values to try for Lasso
    cv_elasticnet_alphas : ndarray, optional
        Alpha values to try for ElasticNet
    cv_l1_ratios : ndarray, optional
        L1 ratio values to try for ElasticNet
        
    Returns:
    --------
    best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio
    """
    # Default parameter grids if not provided
    if cv_lasso_alphas is None:
        cv_lasso_alphas = np.logspace(-6, 1, 100)
    if cv_elasticnet_alphas is None:
        cv_elasticnet_alphas = np.logspace(-6, 1, 100)
    if cv_l1_ratios is None:
        cv_l1_ratios = np.arange(0.01, 1.01, 0.01)
    
    print("Tuning hyperparameters...")
    tuning_start_time = time.time()
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    
    # Tune Lasso
    lasso_cv = LassoCV(alphas=cv_lasso_alphas, cv=tscv, n_jobs=-1, 
                       random_state=42, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train_pct_change)
    best_lasso_alpha = lasso_cv.alpha_
    
    # Tune ElasticNet
    elasticnet_cv = ElasticNetCV(alphas=cv_elasticnet_alphas, l1_ratio=cv_l1_ratios, 
                               cv=tscv, n_jobs=-1, random_state=42, max_iter=10000)
    elasticnet_cv.fit(X_train_scaled, y_train_pct_change)
    best_elasticnet_alpha = elasticnet_cv.alpha_
    best_elasticnet_l1_ratio = elasticnet_cv.l1_ratio_
    
    print(f"Tuning complete ({time.time() - tuning_start_time:.2f}s)")
    print(f"Best Lasso alpha: {best_lasso_alpha:.6f}")
    print(f"Best ElasticNet alpha: {best_elasticnet_alpha:.6f}, l1_ratio: {best_elasticnet_l1_ratio:.2f}")
    
    return best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio

def train_linear_models(X_train_scaled, y_train_pct_change, 
                       best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio):
    """
    Train Lasso and ElasticNet models with tuned hyperparameters.
    
    Parameters:
    -----------
    X_train_scaled : ndarray
        Scaled training features
    y_train_pct_change : Series
        Target percentage change values
    best_lasso_alpha : float
        Best alpha value for Lasso
    best_elasticnet_alpha : float
        Best alpha value for ElasticNet
    best_elasticnet_l1_ratio : float
        Best L1 ratio value for ElasticNet
        
    Returns:
    --------
    lasso_model, elasticnet_model
    """
    print("Training final models...")
    lasso_model = Lasso(alpha=best_lasso_alpha, random_state=42, max_iter=10000)
    elasticnet_model = ElasticNet(alpha=best_elasticnet_alpha, 
                                 l1_ratio=best_elasticnet_l1_ratio, 
                                 random_state=42, max_iter=10000)
    
    lasso_model.fit(X_train_scaled, y_train_pct_change)
    elasticnet_model.fit(X_train_scaled, y_train_pct_change)
    print("Models trained successfully")
    
    return lasso_model, elasticnet_model

def iterative_forecast(model, last_known_level, current_features_scaled_df, 
                      feature_cols, feature_means, feature_stds, 
                      target_pct_change_col, lag_numbers, steps):
    """
    Generate multi-step forecasts by iteratively updating features.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict method
    last_known_level : float
        Last known actual level
    current_features_scaled_df : DataFrame
        Current scaled features
    feature_cols : list
        List of feature column names
    feature_means : dict
        Feature means for scaling
    feature_stds : dict
        Feature standard deviations for scaling
    target_pct_change_col : str
        Name of the target percentage change column
    lag_numbers : dict
        Dictionary mapping feature names to lag numbers
    steps : int
        Number of steps to forecast
        
    Returns:
    --------
    predicted_levels : list
        List of predicted levels
    predicted_pct_changes : list
        List of predicted percentage changes
    """
    predicted_levels = []
    predicted_pct_changes = []
    current_pred_level = last_known_level
    
    for step in range(steps):
        # Predict 1 step ahead based on current features
        pred_pct = model.predict(current_features_scaled_df.values)[0]
        predicted_pct_changes.append(pred_pct)
        
        # Calculate predicted level
        current_pred_level = current_pred_level * (1 + pred_pct)
        predicted_levels.append(current_pred_level)
        
        if step < steps - 1:
            # Unscale current features
            current_features_unscaled = {}
            for col in feature_cols:
                current_features_unscaled[col] = (current_features_scaled_df[col].values[0] * 
                                                  feature_stds[col] + 
                                                  feature_means[col])
            
            # Update lagged target features
            for feature, lag in lag_numbers.items():
                if lag == 1:
                    # Lag 1 uses the prediction we just made
                    current_features_unscaled[feature] = pred_pct
                else:
                    # Lag > 1 uses the value from the feature with lag-1
                    prev_lag_feature = feature.replace(f"_lag{lag}", f"_lag{lag-1}")
                    if prev_lag_feature in feature_cols:
                        current_features_unscaled[feature] = current_features_unscaled[prev_lag_feature]
                    else:
                        print(f"Warning: Could not find feature {prev_lag_feature} to update {feature}")
            
            # Scale the updated features
            next_features_scaled = {}
            for col in feature_cols:
                next_features_scaled[col] = ((current_features_unscaled[col] - 
                                             feature_means[col]) / 
                                             feature_stds[col])
            
            # Update features for next step
            current_features_scaled_df = pd.DataFrame([next_features_scaled], 
                                                    columns=feature_cols)
    
    return predicted_levels, predicted_pct_changes

def walk_forward_validation_linear(X_train_initial, y_train_pct_change_initial, y_train_level_initial,
                                  X_test, y_test_pct_change, y_test_level,
                                  feature_cols, target_pct_change_col,
                                  max_horizon=8, n_cv_splits=9):
    """
    Perform walk-forward validation for Lasso and ElasticNet models.
    
    Parameters:
    -----------
    X_train_initial : DataFrame
        Initial training features
    y_train_pct_change_initial : Series
        Initial training target percentage changes
    y_train_level_initial : Series
        Initial training target levels
    X_test : DataFrame
        Test features
    y_test_pct_change : Series
        Test target percentage changes
    y_test_level : Series
        Test target levels
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target percentage change column
    max_horizon : int
        Maximum forecast horizon
    n_cv_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    all_predictions : dict
        Dictionary of predictions for each horizon
    results_summary : list
        List of performance metrics by horizon
    """
    # Identify lagged features
    house_index_features, lag_numbers, exogenous_features = identify_lagged_features(
        feature_cols, target_pct_change_col)
    
    # Store predictions for each horizon
    all_predictions = {h: {'lasso': [], 'elasticnet': [], 'actual': [], 'last_known_level': []} 
                      for h in range(1, max_horizon + 1)}
    test_indices = X_test.index
    
    # Default hyperparameter grids
    cv_lasso_alphas = np.logspace(-6, 1, 100)
    cv_elasticnet_alphas = np.logspace(-6, 1, 100)
    cv_l1_ratios = np.arange(0.01, 1.01, 0.01)
    
    start_time_wf = time.time()
    print(f"Starting walk-forward validation for {len(X_test) - max_horizon + 1} steps...")
    
    # Outer loop: Iterate through each point in the test set
    for i in range(len(X_test) - max_horizon + 1):
        current_test_date = test_indices[i]
        wf_step_start_time = time.time()
        
        print(f"\nWalk-Forward Step {i+1}/{len(X_test) - max_horizon + 1}: "
              f"Predicting from {current_test_date.strftime('%Y-%m-%d')}...")
        
        # Define current training window (expanding window)
        current_train_end_date = test_indices[i-1] if i > 0 else X_train_initial.index.max()
        X_train_current = pd.concat([X_train_initial, X_test.iloc[:i]])
        y_train_pct_change_current = pd.concat([y_train_pct_change_initial, y_test_pct_change.iloc[:i]])
        y_train_level_current = pd.concat([y_train_level_initial, y_test_level.iloc[:i]])
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_current)
        X_test_step_scaled = scaler.transform(X_test.iloc[[i]])
        
        # Convert scaled data back to DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, 
                                        index=X_train_current.index, 
                                        columns=X_train_current.columns)
        
        # Store scaler's means and stds
        feature_means = dict(zip(feature_cols, scaler.mean_))
        feature_stds = dict(zip(feature_cols, scaler.scale_))
        # Handle zero std dev
        for k, v in feature_stds.items():
            if v == 0:
                print(f"Warning: Feature '{k}' has zero standard deviation. Replacing std with 1.")
                feature_stds[k] = 1.0
        
        # Hyperparameter tuning
        best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio = tune_linear_models(
            X_train_scaled, y_train_pct_change_current, n_cv_splits,
            cv_lasso_alphas, cv_elasticnet_alphas, cv_l1_ratios)
        
        # Train models
        lasso_model, elasticnet_model = train_linear_models(
            X_train_scaled, y_train_pct_change_current,
            best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio)
        
        # Prepare for iterative multi-step prediction
        print(f"   Performing {max_horizon}-step iterative prediction...")
        last_known_actual_level = y_train_level_current.iloc[-1]
        
        # Initialize feature sets for iterative updates
        current_lasso_features_scaled_df = pd.DataFrame(X_test_step_scaled, 
                                                      index=[current_test_date], 
                                                      columns=feature_cols)
        current_elasticnet_features_scaled_df = current_lasso_features_scaled_df.copy()
        
        # Track predicted levels
        lasso_last_pred_level = last_known_actual_level
        elasticnet_last_pred_level = last_known_actual_level
        
        # Store the sequence of pct_change predictions
        lasso_pct_preds_sequence = []
        elasticnet_pct_preds_sequence = []
        
        for step in range(1, max_horizon + 1):
            # Predict 1 step ahead based on current features
            lasso_pred_pct = lasso_model.predict(current_lasso_features_scaled_df.values)[0]
            elasticnet_pred_pct = elasticnet_model.predict(current_elasticnet_features_scaled_df.values)[0]
            
            lasso_pct_preds_sequence.append(lasso_pred_pct)
            elasticnet_pct_preds_sequence.append(elasticnet_pred_pct)
            
            # Calculate predicted level for this step
            lasso_pred_level = lasso_last_pred_level * (1 + lasso_pred_pct)
            elasticnet_pred_level = elasticnet_last_pred_level * (1 + elasticnet_pred_pct)
            
            # Store the prediction for this horizon
            try:
                actual_level_for_step = y_test_level.iloc[i + step - 1]
                all_predictions[step]['lasso'].append(lasso_pred_level)
                all_predictions[step]['elasticnet'].append(elasticnet_pred_level)
                all_predictions[step]['actual'].append(actual_level_for_step)
                all_predictions[step]['last_known_level'].append(last_known_actual_level)
            except IndexError:
                print(f"Warning: Insufficient actual data for step {step}. Skipping.")
                continue
            
            # Feature update for the next step's prediction
            if step < max_horizon:
                # Update lagged features for Lasso
                lasso_next_features_unscaled = {}
                for col in feature_cols:
                    lasso_next_features_unscaled[col] = (current_lasso_features_scaled_df[col].values[0] * 
                                                        feature_stds[col] + 
                                                        feature_means[col])
                
                for feature, lag in lag_numbers.items():
                    if lag == 1:
                        lasso_next_features_unscaled[feature] = lasso_pred_pct
                    else:
                        prev_lag_feature = feature.replace(f"_lag{lag}", f"_lag{lag-1}")
                        if prev_lag_feature in feature_cols:
                            lasso_next_features_unscaled[feature] = lasso_next_features_unscaled[prev_lag_feature]
                
                lasso_next_features_scaled = {}
                for col in feature_cols:
                    lasso_next_features_scaled[col] = ((lasso_next_features_unscaled[col] - 
                                                       feature_means[col]) / 
                                                       feature_stds[col])
                
                # Update lagged features for ElasticNet
                elasticnet_next_features_unscaled = {}
                for col in feature_cols:
                    elasticnet_next_features_unscaled[col] = (current_elasticnet_features_scaled_df[col].values[0] * 
                                                             feature_stds[col] + 
                                                             feature_means[col])
                
                for feature, lag in lag_numbers.items():
                    if lag == 1:
                        elasticnet_next_features_unscaled[feature] = elasticnet_pred_pct
                    else:
                        prev_lag_feature = feature.replace(f"_lag{lag}", f"_lag{lag-1}")
                        if prev_lag_feature in feature_cols:
                            elasticnet_next_features_unscaled[feature] = elasticnet_next_features_unscaled[prev_lag_feature]
                
                elasticnet_next_features_scaled = {}
                for col in feature_cols:
                    elasticnet_next_features_scaled[col] = ((elasticnet_next_features_unscaled[col] - 
                                                           feature_means[col]) / 
                                                           feature_stds[col])
                
                # Update feature DataFrames for next iteration
                current_lasso_features_scaled_df = pd.DataFrame([lasso_next_features_scaled], 
                                                              index=[current_test_date], 
                                                              columns=feature_cols)
                current_elasticnet_features_scaled_df = pd.DataFrame([elasticnet_next_features_scaled], 
                                                                   index=[current_test_date], 
                                                                   columns=feature_cols)
                
                # Update last predicted levels
                lasso_last_pred_level = lasso_pred_level
                elasticnet_last_pred_level = elasticnet_pred_level
        
        step_time = time.time() - wf_step_start_time
        print(f"   Walk-forward step {i+1} completed in {step_time:.2f}s")
    
    total_wf_time = time.time() - start_time_wf
    print(f"\nWalk-Forward Validation complete. Total time: {total_wf_time:.2f} seconds.")
    
    # Calculate metrics for each horizon
    results_summary = []
    for h in range(1, max_horizon + 1):
        preds_lasso = np.array(all_predictions[h]['lasso'])
        preds_elasticnet = np.array(all_predictions[h]['elasticnet'])
        actuals = np.array(all_predictions[h]['actual'])
        last_knowns = np.array(all_predictions[h]['last_known_level'])
        
        if len(actuals) == 0:
            print(f"No predictions for horizon {h}")
            continue
        
        # Calculate metrics
        lasso_metrics = calculate_metrics(actuals, preds_lasso, last_knowns)
        elasticnet_metrics = calculate_metrics(actuals, preds_elasticnet, last_knowns)
        
        results_summary.append({
            'Horizon': h,
            'Lasso_MSE': lasso_metrics['MSE'], 
            'Lasso_RMSE': lasso_metrics['RMSE'], 
            'Lasso_MAE': lasso_metrics['MAE'], 
            'Lasso_MAPE': lasso_metrics['MAPE'], 
            'Lasso_Dir_Acc': lasso_metrics['Dir_Acc'],
            'ElasticNet_MSE': elasticnet_metrics['MSE'], 
            'ElasticNet_RMSE': elasticnet_metrics['RMSE'], 
            'ElasticNet_MAE': elasticnet_metrics['MAE'], 
            'ElasticNet_MAPE': elasticnet_metrics['MAPE'], 
            'ElasticNet_Dir_Acc': elasticnet_metrics['Dir_Acc']
        })
    
    return all_predictions, results_summary

def final_linear_forecast(X_train_final, y_train_pct_change_final, y_train_level_final,
                        feature_cols, target_pct_change_col,
                        num_forecast_steps=8, n_cv_splits=9):
    """
    Generate final forecast using Lasso and ElasticNet models.
    
    Parameters:
    -----------
    X_train_final : DataFrame
        Final training features
    y_train_pct_change_final : Series
        Final training target percentage changes
    y_train_level_final : Series
        Final training target levels
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target percentage change column
    num_forecast_steps : int
        Number of steps to forecast
    n_cv_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    forecast_df : DataFrame
        DataFrame with forecasted values
    """
    # Identify lagged features
    house_index_features, lag_numbers, exogenous_features = identify_lagged_features(
        feature_cols, target_pct_change_col)
    
    # Final scaling
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    
    # Store means/stds for scaling
    feature_means = dict(zip(feature_cols, final_scaler.mean_))
    feature_stds = dict(zip(feature_cols, final_scaler.scale_))
    # Handle zero std dev
    for k, v in feature_stds.items():
        if v == 0:
            print(f"Warning: Feature '{k}' has zero standard deviation. Replacing std with 1.")
            feature_stds[k] = 1.0
    
    # Hyperparameter tuning
    best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio = tune_linear_models(
        X_train_final_scaled, y_train_pct_change_final, n_cv_splits)
    
    # Train models
    lasso_model, elasticnet_model = train_linear_models(
        X_train_final_scaled, y_train_pct_change_final,
        best_lasso_alpha, best_elasticnet_alpha, best_elasticnet_l1_ratio)
    
    # Prepare for forecasting
    last_known_level = y_train_level_final.iloc[-1]
    last_date = y_train_level_final.index[-1]
    
    # Get the features corresponding to the last date
    last_features_row = X_train_final.iloc[[-1]]
    # Scale these features
    current_features_scaled = final_scaler.transform(last_features_row)
    current_features_scaled_df = pd.DataFrame(current_features_scaled, 
                                            index=[last_date], 
                                            columns=feature_cols)
    
    print(f"Forecasting starting after {last_date.strftime('%Y-%m-%d')} "
          f"using level {last_known_level:.4f}")
    
    # Prepare for forecasting
    forecast_results = []
    
    # Determine date frequency for generating future dates
    inferred_freq = pd.infer_freq(X_train_final.index)
    print(f"Inferred data frequency: {inferred_freq}")
    if inferred_freq is None:
        print("Warning: Could not infer frequency. Assuming Monthly Start ('MS').")
        date_offset = pd.tseries.offsets.DateOffset(months=1)
    else:
        date_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    
    # Generate forecasts
    lasso_levels, lasso_pct_changes = iterative_forecast(
        lasso_model, last_known_level, current_features_scaled_df,
        feature_cols, feature_means, feature_stds,
        target_pct_change_col, lag_numbers, num_forecast_steps)
    
    elasticnet_levels, elasticnet_pct_changes = iterative_forecast(
        elasticnet_model, last_known_level, current_features_scaled_df,
        feature_cols, feature_means, feature_stds,
        target_pct_change_col, lag_numbers, num_forecast_steps)
    
    # Create forecast DataFrame
    current_pred_date = last_date
    for step in range(num_forecast_steps):
        current_pred_date = current_pred_date + date_offset
        forecast_date_str = current_pred_date.strftime('%Y-%m-%d')
        
        forecast_results.append({
            'Date': forecast_date_str,
            'Lasso_Forecast': lasso_levels[step],
            'ElasticNet_Forecast': elasticnet_levels[step]
        })
    
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df, lasso_model, elasticnet_model

def visualize_linear_feature_importance(lasso_model, elasticnet_model, feature_cols):
    """
    Visualize feature importance for Lasso and ElasticNet models.
    
    Parameters:
    -----------
    lasso_model : Lasso
        Trained Lasso model
    elasticnet_model : ElasticNet
        Trained ElasticNet model
    feature_cols : list
        List of feature column names
    """
    # Extract coefficients
    lasso_coef = lasso_model.coef_
    elasticnet_coef = elasticnet_model.coef_
    
    # Ensure number of coefficients matches number of features
    if len(lasso_coef) != len(feature_cols) or len(elasticnet_coef) != len(feature_cols):
        raise ValueError("Mismatch between number of coefficients and feature names")
    
    # Create importance DataFrames
    lasso_importance = pd.Series(np.abs(lasso_coef), index=feature_cols).sort_values(ascending=False)
    elasticnet_importance = pd.Series(np.abs(elasticnet_coef), index=feature_cols).sort_values(ascending=False)
    
    # Filter out zero importance
    lasso_importance_filtered = lasso_importance[lasso_importance > 1e-6]
    elasticnet_importance_filtered = elasticnet_importance[elasticnet_importance > 1e-6]
    
    # Create plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle('Feature Importance (Absolute Coefficient Values)', fontsize=16)
    
    # Plot Lasso Importances
    if not lasso_importance_filtered.empty:
        ax1 = axes[0]
        lasso_importance_filtered.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Lasso Model')
        ax1.invert_yaxis()
        ax1.set_xlabel('Absolute Coefficient Value')
        ax1.grid(axis='x', linestyle='--', alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, 'Lasso selected no features', ha='center', va='center', fontsize=12)
        axes[0].set_title('Lasso Model')
        axes[0].set_yticks([])
    
    # Plot ElasticNet Importances
    if not elasticnet_importance_filtered.empty:
        ax2 = axes[1]
        elasticnet_importance_filtered.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('ElasticNet Model')
        ax2.invert_yaxis()
        ax2.set_xlabel('Absolute Coefficient Value')
        ax2.grid(axis='x', linestyle='--', alpha=0.6)
    else:
        axes[1].text(0.5, 0.5, 'ElasticNet selected no features', ha='center', va='center', fontsize=12)
        axes[1].set_title('ElasticNet Model')
        axes[1].set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# =============================================================================
# SVR Model Functions
# =============================================================================

# SVR tuning function correction
def tune_svr_model(X_train_scaled, y_train_pct_change, n_cv_splits=9):
    """
    Tune SVR model using random search.
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    print("Tuning SVR hyperparameters...")
    tuning_start_time = time.time()
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    
    # SVR Parameter Distributions for RandomizedSearch - EXACT match with original
    svr_param_dist = {
        'C': np.arange(1, 101, 0.1),
        'epsilon': uniform(loc=0.0001, scale=0.1)
    }
    
    # SVR model with linear kernel
    svr_linear = SVR(kernel='linear', max_iter=200000)
    
    # RandomizedSearchCV - EXACT match with original iterations
    random_search = RandomizedSearchCV(
        estimator=svr_linear,
        param_distributions=svr_param_dist,
        n_iter=100000,  # IMPORTANT: match original 100000 iterations
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train_scaled, y_train_pct_change)
    
    # Get best params
    best_params = random_search.best_params_
    best_svr_c = best_params['C']
    best_svr_epsilon = best_params['epsilon']
    
    print(f"Tuning complete ({time.time() - tuning_start_time:.2f}s)")
    print(f"Best SVR C: {best_svr_c:.5f}")
    print(f"Best SVR epsilon: {best_svr_epsilon:.5f}")
    
    return best_svr_c, best_svr_epsilon

def train_svr_model(X_train_scaled, y_train_pct_change, best_svr_c, best_svr_epsilon):
    """
    Train SVR model with tuned hyperparameters.
    
    Parameters:
    -----------
    X_train_scaled : ndarray
        Scaled training features
    y_train_pct_change : Series
        Target percentage change values
    best_svr_c : float
        Best C value for SVR
    best_svr_epsilon : float
        Best epsilon value for SVR
        
    Returns:
    --------
    svr_model
    """
    print("Training final SVR model...")
    svr_model = SVR(kernel='linear', C=best_svr_c, epsilon=best_svr_epsilon, max_iter=200000)
    svr_model.fit(X_train_scaled, y_train_pct_change)
    print("SVR model trained successfully")
    
    return svr_model

def walk_forward_validation_svr(X_train_initial, y_train_pct_change_initial, y_train_level_initial,
                              X_test, y_test_pct_change, y_test_level,
                              feature_cols, target_pct_change_col,
                              max_horizon=8, n_cv_splits=5):
    """
    Perform walk-forward validation for SVR model.
    
    Parameters:
    -----------
    X_train_initial : DataFrame
        Initial training features
    y_train_pct_change_initial : Series
        Initial training target percentage changes
    y_train_level_initial : Series
        Initial training target levels
    X_test : DataFrame
        Test features
    y_test_pct_change : Series
        Test target percentage changes
    y_test_level : Series
        Test target levels
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target percentage change column
    max_horizon : int
        Maximum forecast horizon
    n_cv_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    all_predictions_svr : dict
        Dictionary of predictions for each horizon
    results_summary_svr : list
        List of performance metrics by horizon
    """
    # Identify lagged features
    house_index_features, lag_numbers, exogenous_features = identify_lagged_features(
        feature_cols, target_pct_change_col)
    
    # Store predictions for each horizon
    all_predictions_svr = {h: {'preds': [], 'actual': [], 'last_known_level': []} 
                         for h in range(1, max_horizon + 1)}
    test_indices = X_test.index
    
    # Initial scaling
    initial_scaler = StandardScaler()
    X_train_initial_scaled = initial_scaler.fit_transform(X_train_initial)
    
    # Store scaler attributes for consistent use
    initial_scaler_means = initial_scaler.mean_
    initial_scaler_stds = initial_scaler.scale_
    initial_scaler_stds = np.where(initial_scaler_stds == 0, 1.0, initial_scaler_stds)
    
    # Hyperparameter tuning (performed ONCE)
    print("Tuning SVR hyperparameters once for all walk-forward steps...")
    best_svr_c, best_svr_epsilon = tune_svr_model(
        X_train_initial_scaled, y_train_pct_change_initial, n_cv_splits)
    print("Using these fixed parameters for all walk-forward steps")
    
    # Walk-forward validation
    start_time_wf = time.time()
    
    # Initialize history
    history_X = X_train_initial.copy()
    history_y_pct_change = y_train_pct_change_initial.copy()
    history_y_level = y_train_level_initial.copy()
    
    for i in range(len(X_test) - max_horizon + 1):
        current_test_date = test_indices[i]
        wf_step_start_time = time.time()
        
        print(f"\nWalk-Forward Step {i+1}/{len(X_test) - max_horizon + 1}: "
              f"Predicting from {current_test_date.strftime('%Y-%m-%d')}...")
        
        # Define current training window
        X_train_current = history_X
        y_train_pct_change_current = history_y_pct_change
        y_train_level_current = history_y_level
        
        # Scale current training data using initial scaler
        X_train_current_scaled = initial_scaler.transform(X_train_current)
        
        # Train SVR model with fixed hyperparameters
        svr_model = SVR(kernel='linear', C=best_svr_c, epsilon=best_svr_epsilon, max_iter=200000)
        svr_model.fit(X_train_current_scaled, y_train_pct_change_current)
        
        # Iterative multi-step prediction
        last_known_actual_level = y_train_level_current.iloc[-1]
        first_step_features_raw = X_test.iloc[[i]]
        current_features_scaled_np = initial_scaler.transform(first_step_features_raw)
        current_pred_level = last_known_actual_level
        
        for step in range(1, max_horizon + 1):
            svr_pred_pct = svr_model.predict(current_features_scaled_np)[0]
            current_pred_level = current_pred_level * (1 + svr_pred_pct)
            
            try:
                actual_level_for_step = y_test_level.iloc[i + step - 1]
                all_predictions_svr[step]['preds'].append(current_pred_level)
                all_predictions_svr[step]['actual'].append(actual_level_for_step)
                all_predictions_svr[step]['last_known_level'].append(last_known_actual_level)
            except IndexError:
                print(f"Warning: Insufficient actual data for step {step}. Skipping.")
                continue
            
            if step < max_horizon:
                # Unscale and update features for next step
                current_features_unscaled = (current_features_scaled_np * initial_scaler_stds) + initial_scaler_means
                current_features_unscaled_dict = dict(zip(feature_cols, current_features_unscaled[0]))
                next_features_unscaled_dict = current_features_unscaled_dict.copy()
                
                # Update lagged features
                for lag in range(max(lag_numbers.values()), 0, -1):
                    matching_features = [f for f, l in lag_numbers.items() if l == lag]
                    for current_lag_col in matching_features:
                        if lag == 1:
                            next_features_unscaled_dict[current_lag_col] = svr_pred_pct
                        else:
                            prev_lag_col = current_lag_col.replace(f"_lag{lag}", f"_lag{lag-1}")
                            if prev_lag_col in feature_cols:
                                next_features_unscaled_dict[current_lag_col] = current_features_unscaled_dict[prev_lag_col]
                
                next_features_unscaled_np = np.array([next_features_unscaled_dict[name] for name in feature_cols]).reshape(1, -1)
                current_features_scaled_np = initial_scaler.transform(next_features_unscaled_np)
        
        # Update history
        history_X = pd.concat([history_X, X_test.iloc[[i]]])
        history_y_pct_change = pd.concat([history_y_pct_change, y_test_pct_change.iloc[[i]]])
        history_y_level = pd.concat([history_y_level, y_test_level.iloc[[i]]])
        
        step_time = time.time() - wf_step_start_time
        print(f"   Step {i + 1} finished in {step_time:.2f} seconds.")
    
    total_wf_time = time.time() - start_time_wf
    print(f"\nSVR Walk-Forward Validation complete. Total time: {total_wf_time:.2f} seconds.")
    
    # Calculate metrics for each horizon
    results_summary_svr = []
    for h in range(1, max_horizon + 1):
        preds = np.array(all_predictions_svr[h]['preds'])
        actuals = np.array(all_predictions_svr[h]['actual'])
        last_knowns = np.array(all_predictions_svr[h]['last_known_level'])
        
        if len(actuals) == 0:
            print(f"No predictions for horizon {h}")
            results_summary_svr.append({
                'Horizon': h, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 
                'MAPE': np.nan, 'Dir_Acc': np.nan
            })
            continue
        
        # Calculate metrics
        svr_metrics = calculate_metrics(actuals, preds, last_knowns)
        
        results_summary_svr.append({
            'Horizon': h,
            'MSE': svr_metrics['MSE'],
            'RMSE': svr_metrics['RMSE'],
            'MAE': svr_metrics['MAE'],
            'MAPE': svr_metrics['MAPE'],
            'Dir_Acc': svr_metrics['Dir_Acc']
        })
    
    return all_predictions_svr, results_summary_svr

def final_svr_forecast(X_train_final, y_train_pct_change_final, y_train_level_final,
                      feature_cols, target_pct_change_col,
                      num_forecast_steps=8, n_cv_splits=9):
    """
    Generate final forecast using SVR model.
    
    Parameters:
    -----------
    X_train_final : DataFrame
        Final training features
    y_train_pct_change_final : Series
        Final training target percentage changes
    y_train_level_final : Series
        Final training target levels
    feature_cols : list
        List of feature column names
    target_pct_change_col : str
        Name of the target percentage change column
    num_forecast_steps : int
        Number of steps to forecast
    n_cv_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    forecast_df : DataFrame
        DataFrame with forecasted values
    svr_model : SVR
        Trained SVR model
    """
    # Identify lagged features
    house_index_features, lag_numbers, exogenous_features = identify_lagged_features(
        feature_cols, target_pct_change_col)
    
    # Final scaling
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    
    # Custom tuning function with 10,000 iterations specifically for final forecast
    print("Tuning SVR hyperparameters for final forecast...")
    tuning_start_time = time.time()
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    
    # SVR Parameter Distributions for RandomizedSearch
    svr_param_dist = {
        'C': np.arange(1, 101, 0.1),
        'epsilon': uniform(loc=0.0001, scale=0.1)
    }
    
    # SVR model with linear kernel
    svr_linear = SVR(kernel='linear', max_iter=200000)
    
    # RandomizedSearchCV with 10,000 iterations for final forecast
    random_search = RandomizedSearchCV(
        estimator=svr_linear,
        param_distributions=svr_param_dist,
        n_iter=10000,  # IMPORTANT: Use 10,000 for final forecast
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train_final_scaled, y_train_pct_change_final)
    
    # Get best params
    best_params = random_search.best_params_
    best_svr_c = best_params['C']
    best_svr_epsilon = best_params['epsilon']
    
    print(f"Final forecast tuning complete ({time.time() - tuning_start_time:.2f}s)")
    print(f"Best SVR C: {best_svr_c:.5f}")
    print(f"Best SVR epsilon: {best_svr_epsilon:.5f}")
    
    # Train model
    svr_model = SVR(kernel='linear', C=best_svr_c, epsilon=best_svr_epsilon, max_iter=200000)
    svr_model.fit(X_train_final_scaled, y_train_pct_change_final)
    
    # Rest of the function remains the same as your original implementation
    
    # Prepare for forecasting
    last_known_level = y_train_level_final.iloc[-1]
    last_date = y_train_level_final.index[-1]
    
    # Get the features for the last date
    last_features_row = X_train_final.iloc[[-1]]
    current_features_scaled_np = final_scaler.transform(last_features_row)
    
    print(f"Forecasting starting after {last_date.strftime('%Y-%m-%d')} "
          f"using level {last_known_level:.4f}")
    
    # Determine date frequency
    inferred_freq = pd.infer_freq(X_train_final.index)
    print(f"Inferred data frequency: {inferred_freq}")
    if inferred_freq is None:
        print("Warning: Could not infer frequency. Assuming Monthly Start ('MS').")
        date_offset = pd.tseries.offsets.DateOffset(months=1)
    else:
        date_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    
    # Generate forecasts
    forecast_results = []
    current_pred_level = last_known_level
    current_pred_date = last_date
    
    for step in range(1, num_forecast_steps + 1):
        # Predict 1 step ahead
        svr_pred_pct = svr_model.predict(current_features_scaled_np)[0]
        
        # Calculate predicted level
        current_pred_level = current_pred_level * (1 + svr_pred_pct)
        
        # Calculate the date for this forecast step
        current_pred_date = current_pred_date + date_offset
        forecast_date_str = current_pred_date.strftime('%Y-%m-%d')
        
        # Store results
        forecast_results.append({
            'Date': forecast_date_str,
            'SVR_Forecast': current_pred_level
        })
        
        if step < num_forecast_steps:
            # Unscale current features
            current_features_unscaled = (current_features_scaled_np * final_scaler.scale_) + final_scaler.mean_
            current_features_unscaled_dict = dict(zip(feature_cols, current_features_unscaled[0]))
            
            # Create dict for next step's unscaled features
            next_features_unscaled_dict = current_features_unscaled_dict.copy()
            
            # Update lagged features
            for lag in range(max(lag_numbers.values()), 0, -1):
                matching_features = [f for f, l in lag_numbers.items() if l == lag]
                for current_lag_col in matching_features:
                    if lag == 1:
                        next_features_unscaled_dict[current_lag_col] = svr_pred_pct
                    else:
                        prev_lag_col = current_lag_col.replace(f"_lag{lag}", f"_lag{lag-1}")
                        if prev_lag_col in feature_cols:
                            next_features_unscaled_dict[current_lag_col] = current_features_unscaled_dict[prev_lag_col]
            
            # Convert updated features back to numpy array
            next_features_unscaled_np = np.array([next_features_unscaled_dict[name] for name in feature_cols]).reshape(1, -1)
            
            # Scale the updated features
            current_features_scaled_np = final_scaler.transform(next_features_unscaled_np)
    
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df, svr_model

# =============================================================================
# Historical Mean Model Functions
# =============================================================================

def historical_mean_forecast(y_train_pct_change, y_train_level, num_forecast_steps=8):
    """
    Generate forecast using historical mean of percentage changes.
    
    Parameters:
    -----------
    y_train_pct_change : Series
        Training target percentage changes
    y_train_level : Series
        Training target levels
    num_forecast_steps : int
        Number of steps to forecast
        
    Returns:
    --------
    forecast_df : DataFrame
        DataFrame with forecasted values
    """
    # Calculate historical mean
    mean_pct_change = y_train_pct_change.mean()
    print(f"Historical mean percentage change: {mean_pct_change:.6f}")
    
    # Get last known level and date
    last_known_level = y_train_level.iloc[-1]
    last_date = y_train_level.index[-1]
    
    # Determine date frequency
    inferred_freq = pd.infer_freq(y_train_level.index)
    print(f"Inferred data frequency: {inferred_freq}")
    if inferred_freq is None:
        print("Warning: Could not infer frequency. Assuming Monthly Start ('MS').")
        date_offset = pd.tseries.offsets.DateOffset(months=1)
    else:
        date_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    
    # Generate forecasts
    forecast_results = []
    current_pred_level = last_known_level
    current_pred_date = last_date
    
    for step in range(1, num_forecast_steps + 1):
        # Calculate predicted level
        current_pred_level = current_pred_level * (1 + mean_pct_change)
        
        # Calculate the date for this forecast step
        current_pred_date = current_pred_date + date_offset
        forecast_date_str = current_pred_date.strftime('%Y-%m-%d')
        
        # Store results
        forecast_results.append({
            'Date': forecast_date_str,
            'Mean_Forecast': current_pred_level
        })
    
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df

def walk_forward_validation_mean(y_train_pct_change_initial, y_train_level_initial,
                               y_test_pct_change, y_test_level,
                               max_horizon=8):
    """
    Perform walk-forward validation for historical mean model.
    
    Parameters:
    -----------
    y_train_pct_change_initial : Series
        Initial training target percentage changes
    y_train_level_initial : Series
        Initial training target levels
    y_test_pct_change : Series
        Test target percentage changes
    y_test_level : Series
        Test target levels
    max_horizon : int
        Maximum forecast horizon
        
    Returns:
    --------
    all_predictions_mean : dict
        Dictionary of predictions for each horizon
    results_summary_mean : list
        List of performance metrics by horizon
    """
    # Store predictions for each horizon
    predictions_level = {h: [] for h in range(1, max_horizon + 1)}
    actuals_level = {h: [] for h in range(1, max_horizon + 1)}
    prev_actuals_level = {h: [] for h in range(1, max_horizon + 1)}
    target_dates_level = {h: [] for h in range(1, max_horizon + 1)}
    
    test_indices = y_test_level.index
    start_time_wf = time.time()
    
    # Initialize history
    history_y_pct_change = y_train_pct_change_initial.copy()
    history_y_level = y_train_level_initial.copy()
    
    # Loop through potential forecast origins in the test set
    num_test_points = len(y_test_level)
    num_origins = num_test_points - max_horizon + 1
    
    if num_origins <= 0:
        raise ValueError(f"Test set too short ({num_test_points} points) for max_horizon={max_horizon}")
    
    print(f"\nStarting walk-forward validation for Historical Mean ({num_origins} origins)...")
    
    for i in range(num_origins):
        current_origin_date = test_indices[i]
        last_hist_date = test_indices[i-1] if i > 0 else y_train_level_initial.index[-1]
        
        print(f"Processing Origin {i + 1}/{num_origins} "
              f"(History Ends: {last_hist_date.strftime('%Y-%m-%d')})...", end='\r')
        
        # Calculate expanding historical mean
        current_mean_pct_change = history_y_pct_change.mean()
        
        # Generate multi-step forecast
        last_known_actual_level = history_y_level.iloc[-1]
        current_pred_level = last_known_actual_level
        
        # Determine date frequency
        freq = pd.infer_freq(history_y_level.index)
        if freq is None:
            offset = pd.DateOffset(months=1)
        else:
            offset = pd.tseries.frequencies.to_offset(freq)
        
        for h in range(1, max_horizon + 1):
            # Forecasted pct change is always the historical mean
            forecast_pct_change = current_mean_pct_change
            
            # Calculate predicted level
            current_pred_level = current_pred_level * (1 + forecast_pct_change)
            
            # Determine target date
            target_date = last_hist_date + (offset * h)
            
            # Store results if target date has actual data
            if target_date in y_test_level.index:
                predictions_level[h].append(current_pred_level)
                target_dates_level[h].append(target_date)
                actuals_level[h].append(y_test_level.loc[target_date])
                
                # Get previous actual for DA calculation
                prev_actual_date = target_date - offset
                if prev_actual_date in y_test_level.index or prev_actual_date in y_train_level_initial.index:
                    if prev_actual_date in y_test_level.index:
                        prev_actuals_level[h].append(y_test_level.loc[prev_actual_date])
                    else:
                        prev_actuals_level[h].append(y_train_level_initial.loc[prev_actual_date])
                else: # Need to pop corresponding prediction/actual if prev is missing
                    predictions_level[h].pop()
                    target_dates_level[h].pop()
                    actuals_level[h].pop()
        
        # Update history for the next outer loop iteration
        history_y_pct_change = pd.concat([history_y_pct_change, y_test_pct_change.iloc[[i]]])
        history_y_level = pd.concat([history_y_level, y_test_level.iloc[[i]]])
    
    total_wf_time = time.time() - start_time_wf
    print(f"\nWalk-forward validation complete. Total time: {total_wf_time:.2f} seconds.")
    
    # Calculate metrics for each horizon
    results_summary_mean = []
    
    for h in range(1, max_horizon + 1):
        preds = np.array(predictions_level[h])
        actuals = np.array(actuals_level[h])
        prev_actuals = np.array(prev_actuals_level[h])
        
        if len(preds) == 0:
            print(f"No predictions for horizon {h}")
            results_summary_mean.append({
                'Horizon': h, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 
                'MAPE': np.nan, 'Dir_Acc': np.nan
            })
            continue
        
        # Ensure lengths match
        min_len = min(len(preds), len(actuals), len(prev_actuals))
        preds = preds[:min_len]
        actuals = actuals[:min_len]
        prev_actuals = prev_actuals[:min_len]
        
        if min_len > 0:
            # Calculate metrics
            mean_metrics = calculate_metrics(actuals, preds, prev_actuals)
            
            results_summary_mean.append({
                'Horizon': h,
                'MSE': mean_metrics['MSE'],
                'RMSE': mean_metrics['RMSE'],
                'MAE': mean_metrics['MAE'],
                'MAPE': mean_metrics['MAPE'],
                'Dir_Acc': mean_metrics['Dir_Acc']
            })
        else:
            results_summary_mean.append({
                'Horizon': h, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 
                'MAPE': np.nan, 'Dir_Acc': np.nan
            })
    
    return predictions_level, actuals_level, prev_actuals_level, target_dates_level, results_summary_mean

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_forecast_comparison(y_train, y_test, model_forecasts, 
                           horizon=1, model_name='Model', 
                           display_years=5, figsize=(15, 7)):
    """
    Plot historical data and forecasts for a specific horizon.
    
    Parameters:
    -----------
    y_train : Series
        Training target levels
    y_test : Series
        Test target levels
    model_forecasts : dict
        Dictionary of forecasts by horizon
    horizon : int
        Forecast horizon to plot
    model_name : str
        Name of the model
    display_years : int
        Number of years of history to display
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Extract forecasts and dates for the specified horizon
    if horizon not in model_forecasts:
        raise ValueError(f"Horizon {horizon} not found in model_forecasts")
    
    forecast_values = model_forecasts[horizon]
    
    # Determine plot range
    plot_start_date = y_test.index.min() - pd.DateOffset(years=display_years)
    plot_end_date = y_test.index.max()
    
    # Plot historical data
    plt.plot(y_train.loc[y_train.index >= plot_start_date].index, 
            y_train.loc[y_train.index >= plot_start_date].values, 
            'k-', label='Training Data', linewidth=1.5, alpha=0.7)
    plt.plot(y_test.index, y_test.values, color='gray', linestyle='--', 
            label='Actual Test Data', linewidth=1.5, alpha=0.7)
    
    # Plot forecasts
    plt.plot(forecast_values['dates'], forecast_values['predictions'], 
            marker='o', linestyle='-', label=f'{model_name} {horizon}-Step Forecast')
    
    # Formatting
    plt.title(f'{model_name}: Actual vs {horizon}-Step Forecasted House Index')
    plt.ylabel('House Index')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(plot_start_date, plot_end_date)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_final_forecast(y_history, forecast_df, model_name='Model', 
                      display_years=5, figsize=(15, 7)):
    """
    Plot historical data and final forecast.
    
    Parameters:
    -----------
    y_history : Series
        Historical target levels
    forecast_df : DataFrame
        DataFrame with forecasted values
    model_name : str
        Name of the model
    display_years : int
        Number of years of history to display
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get last known level and date
    last_known_level = y_history.iloc[-1]
    last_date = y_history.index[-1]
    
    # Determine plot range
    display_start_date = last_date - pd.DateOffset(years=display_years)
    display_end_date = forecast_df.index.max() + pd.DateOffset(months=1)
    
    # Plot historical data
    plt.plot(y_history.loc[y_history.index >= display_start_date].index, 
            y_history.loc[y_history.index >= display_start_date].values, 
            'k-', label='Historical Actual Data', linewidth=1.5)
    
    # Combine last historical point with forecasts for smooth line
    forecast_col = forecast_df.columns[0]  # Assuming first column is the forecast
    plot_index = pd.Index([last_date]).union(forecast_df.index)
    plot_values = pd.concat([pd.Series([last_known_level], index=[last_date]), 
                           forecast_df[forecast_col]])
    
    # Plot forecast
    plt.plot(plot_index, plot_values, 'b-o', label=f'{model_name} Forecast', 
            linewidth=1.5, markersize=4)
    
    # Formatting
    plt.title(f'House Index: Historical Data and {model_name} Forecast after {last_date.strftime("%Y-%m-%d")}', 
             fontsize=14)
    plt.ylabel('House Index', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(display_start_date, display_end_date)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_all_forecasts(y_history, forecast_dfs, models=None, 
                     display_years=5, figsize=(15, 7)):
    """
    Plot historical data and forecasts from multiple models.
    
    Parameters:
    -----------
    y_history : Series
        Historical target levels
    forecast_dfs : list
        List of DataFrames with forecasted values
    models : list
        List of model names
    display_years : int
        Number of years of history to display
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get last known level and date
    last_known_level = y_history.iloc[-1]
    last_date = y_history.index[-1]
    
    # Determine plot range
    display_start_date = last_date - pd.DateOffset(years=display_years)
    display_end_date = max([df.index.max() for df in forecast_dfs]) + pd.DateOffset(months=1)
    
    # Plot historical data
    plt.plot(y_history.loc[y_history.index >= display_start_date].index, 
            y_history.loc[y_history.index >= display_start_date].values, 
            'k-', label='Historical Actual Data', linewidth=1.5)
    
    # Plot forecasts from each model
    colors = ['b', 'g', 'r', 'm', 'c', 'y']
    markers = ['o', 's', '^', 'p', '*', 'x']
    
    for i, forecast_df in enumerate(forecast_dfs):
        model_name = models[i] if models and i < len(models) else f'Model {i+1}'
        forecast_col = forecast_df.columns[0]  # Assuming first column is the forecast
        
        # Combine last historical point with forecasts for smooth line
        plot_index = pd.Index([last_date]).union(forecast_df.index)
        plot_values = pd.concat([pd.Series([last_known_level], index=[last_date]), 
                               forecast_df[forecast_col]])
        
        # Plot forecast
        plt.plot(plot_index, plot_values, 
                f'{colors[i % len(colors)]}-{markers[i % len(markers)]}', 
                label=f'{model_name} Forecast', 
                linewidth=1.5, markersize=6)
    
    # Formatting
    plt.title(f'House Index: Historical Data and Model Forecasts after {last_date.strftime("%Y-%m-%d")}', 
             fontsize=14)
    plt.ylabel('House Index', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(display_start_date, display_end_date)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()