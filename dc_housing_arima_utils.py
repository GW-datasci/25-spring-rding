"""
DC Housing ARIMA Utilities

This module contains helper functions for ARIMA modeling and validation,
specifically tailored for the DC housing and economic indicators dataset analysis.
"""

# Import necessary libraries
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings globally within the utility module if desired,
# or manage them in the notebook that imports this.
# warnings.filterwarnings("ignore")

# ==================================
# Metrics Functions
# ==================================

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Handles potential division by zero by masking zero values in y_true.

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: MAPE value in percentage, or np.nan if calculation is not possible.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Create a mask for non-zero true values
    if not np.any(mask):
        # Handle case where all true values are zero
        return np.nan if np.any(y_pred != 0) else 0.0
    y_true_masked, y_pred_masked = y_true[mask], y_pred[mask]
    # Calculate MAPE using only non-zero true values
    return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100

def directional_accuracy(y_true, y_pred, y_hist):
    """
    Calculate Directional Accuracy.

    Measures the percentage of times the forecast correctly predicts the
    direction of change (up, down, or flat) from the previous actual value.

    Args:
        y_true (array-like): Array of true values for the forecast period.
        y_pred (array-like): Array of predicted values for the forecast period.
        y_hist (array-like): Array of actual values immediately preceding each true value
                             (must be same length as y_true and y_pred).

    Returns:
        float: Directional accuracy percentage, or np.nan if inputs are invalid.
    """
    y_true, y_pred, y_hist = np.array(y_true), np.array(y_pred), np.array(y_hist)
    # Basic validation for input arrays
    if len(y_true) < 1 or len(y_true) != len(y_pred) or len(y_true) != len(y_hist):
        print("Warning: Input arrays for directional accuracy have mismatched lengths or are empty.")
        return np.nan

    true_diff = y_true - y_hist # Actual change from previous period
    pred_diff = y_pred - y_hist # Predicted change from previous period

    sign_true = np.sign(true_diff)
    sign_pred = np.sign(pred_diff)

    # Correct direction if signs match, OR if both actual and predicted changes are zero (no change)
    correct_direction = (sign_true == sign_pred)
    correct_direction = correct_direction | ((true_diff == 0) & (pred_diff == 0))

    return np.mean(correct_direction) * 100

# ==================================
# ARIMA Model Functions
# ==================================

def fit_auto_arima(series, m=4, seasonal=True, stepwise=False, max_p=8, max_q=8,
                   max_P=5, max_Q=5, max_order=20, start_p=1, start_q=1, start_P=0,
                   d=None, D=None, trace=True, error_action='ignore',
                   suppress_warnings=True, **kwargs):
    """
    Fit an Auto ARIMA model using pmdarima.auto_arima.

    Allows specifying parameters for the search, including enabling exhaustive search.

    Args:
        series (pd.Series): The time series data to fit the model on.
        m (int): The seasonal periodicity (e.g., 4 for quarterly, 12 for monthly).
        seasonal (bool): Whether to fit a seasonal model.
        stepwise (bool): Use stepwise algorithm (faster) or exhaustive search (slower).
        max_p, max_q, max_P, max_Q, max_order (int): Limits for ARIMA orders search space.
        start_p, start_q, start_P (int): Starting points for order search.
        d, D (int or None): Orders of non-seasonal and seasonal differencing. If None, auto-detected.
        trace (bool): Print status updates during the model search.
        error_action (str): Action to take if a model fit fails ('ignore', 'warn', 'raise').
        suppress_warnings (bool): Suppress convergence and other warnings from statsmodels.
        **kwargs: Additional arguments passed directly to pm.auto_arima.

    Returns:
        pmdarima.arima.ARIMA: The best fitted Auto ARIMA model found.
    """
    print(f"  Fitting AutoARIMA (m={m}, seasonal={seasonal}, stepwise={stepwise})...")
    model = pm.auto_arima(
        series,
        start_p=start_p, start_q=start_q,
        max_p=max_p, max_q=max_q,
        m=m,
        start_P=start_P, max_P=max_P, max_Q=max_Q,
        seasonal=seasonal,
        d=d, D=D,
        max_order=max_order,
        trace=trace,
        error_action=error_action,
        suppress_warnings=suppress_warnings,
        stepwise=stepwise,
        **kwargs # Pass any other overrides or extra args
    )
    return model

# ==================================
# Plotting Functions
# ==================================

def plot_forecast(series, model, forecast_steps=8, start_date=None, title=None):
    """
    Generate and plot forecasts with 95% confidence intervals for a given model and series.

    Args:
        series (pd.Series): The original time series data (including history).
        model (pmdarima.arima.ARIMA): The fitted ARIMA model.
        forecast_steps (int): Number of steps ahead to forecast.
        start_date (pd.Timestamp or str, optional): Date from which to start plotting historical data.
                                                    Defaults to '2020-01-01'.
        title (str, optional): Title for the plot. Defaults to a generated title.

    Returns:
        pd.DataFrame: DataFrame containing the point forecasts and confidence intervals.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        print("Error: Series index must be a DatetimeIndex for forecasting.")
        return None

    last_date = series.index.max()
    print(f"  Generating {forecast_steps}-step ahead forecast from {last_date.strftime('%Y-%m-%d')}...")

    try:
        # Generate forecasts with confidence intervals
        point_forecasts, conf_int = model.predict(n_periods=forecast_steps,
                                                  return_conf_int=True,
                                                  alpha=0.05) # alpha=0.05 for 95% CI

        # Create forecast dataframe with the correct future index
        # Infer frequency if possible, otherwise assume quarterly start ('QS')
        freq = pd.infer_freq(series.index) or 'QS'
        print(f"  Inferred/Assumed frequency for forecast index: {freq}")
        # Calculate the start date for the forecast index
        forecast_start_date = last_date + pd.tseries.frequencies.to_offset(freq)

        forecast_index = pd.date_range(start=forecast_start_date,
                                       periods=forecast_steps, freq=freq)

        forecast_df = pd.DataFrame({
            'Forecast': point_forecasts,
            'Lower 95% CI': conf_int[:, 0],
            'Upper 95% CI': conf_int[:, 1]
        }, index=forecast_index)

        # --- Visualization ---
        plt.figure(figsize=(14, 7))

        # Determine the start date for plotting historical data
        viz_start_date = pd.to_datetime(start_date) if start_date else pd.Timestamp('2020-01-01')
        viz_series = series[series.index >= viz_start_date]

        # Plot historical data
        plt.plot(viz_series.index, viz_series.values, label='Historical Data', color='blue')

        # Plot point forecasts
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange', marker='o')

        # Plot confidence interval
        plt.fill_between(forecast_df.index,
                         forecast_df['Lower 95% CI'],
                         forecast_df['Upper 95% CI'],
                         color='gray', alpha=0.3, label='95% Prediction Interval')

        # Set plot properties
        plot_title = title or f'{series.name or "Series"} Forecast ({forecast_steps}-Step Ahead with 95% PI)'
        plt.title(plot_title)
        plt.xlabel('Date')
        plt.ylabel(series.name if series.name else 'Value')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Adjust x-axis limits to show some history and the full forecast
        plot_end_date = forecast_df.index[-1] + pd.DateOffset(months=6) # Add buffer
        plt.xlim(left=viz_start_date, right=plot_end_date)

        plt.tight_layout()
        plt.show()

        return forecast_df

    except Exception as e:
        print(f"Error during forecast generation or plotting: {e}")
        return None


def plot_walk_forward_results(column_name, train_data, test_data, predictions, prediction_indices,
                              steps_to_plot=[1, 4, 8], viz_start_date=None):
    """
    Plot walk-forward validation results for a single column.

    Args:
        column_name (str): Name of the column being plotted.
        train_data (pd.Series): Training data portion used before the test set started.
        test_data (pd.Series): Test data (actuals) portion.
        predictions (dict): Dictionary where keys are step numbers (int) and values are lists of forecasts.
        prediction_indices (dict): Dictionary where keys are step numbers (int) and values are lists of pd.Timestamp indices corresponding to predictions.
        steps_to_plot (list): List of step numbers (e.g., [1, 4, 8]) to include in the plot.
        viz_start_date (pd.Timestamp or str, optional): Date to start the visualization from. Defaults to '2015-01-01'.
    """
    fig, ax = plt.subplots(figsize=(15, 7))

    # Determine visualization start date
    viz_start_date_ts = pd.to_datetime(viz_start_date) if viz_start_date else pd.Timestamp('2015-01-01')

    # Plot relevant portion of training data
    train_to_plot = train_data[train_data.index >= viz_start_date_ts]
    if not train_to_plot.empty:
         ax.plot(train_to_plot.index, train_to_plot.values, label='Training Data', color='grey', linestyle='--')

    # Plot actual test data
    ax.plot(test_data.index, test_data.values, label='Test Data (Actuals)', color='blue', linewidth=2)

    # Plot forecasts for selected steps
    colors = {1: 'orange', 4: 'green', 8: 'red'} # Colors for different forecast horizons
    for step in steps_to_plot:
         if step in predictions and step in prediction_indices and len(prediction_indices[step]) > 0:
              # Create a pandas Series for easier plotting with dates
              # Ensure indices are sorted if they came out of order (unlikely but possible)
              pred_series = pd.Series(data=predictions[step], index=prediction_indices[step]).sort_index()
              ax.plot(pred_series.index, pred_series.values,
                      label=f'{step}-Step Forecast',
                      color=colors.get(step, 'black'), # Default to black if step not in colors dict
                      marker='o', linestyle=':', markersize=4)
         else:
              print(f"  Note: No data to plot for {step}-step forecast.")

    # Configure plot aesthetics
    ax.set_title(f'Walk-Forward Validation: {column_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set x-axis limits to focus on the relevant period
    plot_end_date = test_data.index.max() + pd.DateOffset(months=6) # Add buffer
    ax.set_xlim(left=viz_start_date_ts, right=plot_end_date)

    plt.tight_layout()
    plt.show()


# ==================================
# Validation Functions
# ==================================

def perform_walk_forward_validation(series, order, seasonal_order, n_test_quarters=36,
                                    steps_to_calculate=range(1, 9)):
    """
    Perform walk-forward validation for a single time series using fixed model orders.

    Refits the model at each step of the test period.

    Args:
        series (pd.Series): The full time series data (must be float type, NaNs dropped).
        order (tuple): The ARIMA order parameters (p,d,q).
        seasonal_order (tuple): The seasonal ARIMA order parameters (P,D,Q,m).
        n_test_quarters (int): Number of quarters to use for the test set.
        steps_to_calculate (range or list): Forecast horizons (e.g., 1-step, 2-step...)
                                             to generate predictions for.

    Returns:
        tuple: A tuple containing four dictionaries:
            - predictions: {step: [forecast_values]}
            - actuals: {step: [actual_values]}
            - history: {step: [value_before_forecast]}
            - prediction_indices: {step: [forecast_timestamps]}
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        print("Error: Series index must be a DatetimeIndex for walk-forward validation.")
        return {}, {}, {}, {}

    # Split data into initial training set and test set
    train_data_initial = series[:-n_test_quarters]
    test_data = series[-n_test_quarters:]

    print(f"  Validation Train Period: {train_data_initial.index.min().strftime('%Y-%m-%d')} to {train_data_initial.index.max().strftime('%Y-%m-%d')}")
    print(f"  Validation Test Period : {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")

    # Prepare storage for multi-step predictions
    max_steps = max(steps_to_calculate)
    predictions = {step: [] for step in steps_to_calculate}
    actuals = {step: [] for step in steps_to_calculate}
    history = {step: [] for step in steps_to_calculate}
    prediction_indices = {step: [] for step in steps_to_calculate}

    # Walk forward through the test period
    print(f"  Starting walk-forward loop for {len(test_data)} test points...")
    for i in range(len(test_data)):
        current_train_end_index = len(train_data_initial) + i
        current_train = series[:current_train_end_index] # Data up to the point before prediction

        # Determine how many steps ahead we can forecast from this point
        steps_possible = len(test_data) - i
        current_max_forecast = min(max_steps, steps_possible)
        if current_max_forecast <= 0: continue # No more steps needed

        try:
            # Refit model with current training data using the provided fixed orders
            refitted_model = pm.ARIMA(order=order,
                                     seasonal_order=seasonal_order,
                                     suppress_warnings=True).fit(current_train)

            # Generate forecasts for the required number of steps
            forecasts = refitted_model.predict(n_periods=current_max_forecast, return_conf_int=False)
            last_actual_val = current_train.iloc[-1] # The value right before the forecast starts

            # Store predictions and actuals for each required step ahead
            for step in steps_to_calculate:
                if step <= current_max_forecast:
                    forecast_val = forecasts[step - 1] # Forecasts array is 0-indexed
                    actual_val_index = current_train_end_index + step - 1 # Index in the original series
                    actual_val = series.iloc[actual_val_index]
                    actual_val_date = series.index[actual_val_index]

                    predictions[step].append(forecast_val)
                    actuals[step].append(actual_val)
                    history[step].append(last_actual_val)
                    prediction_indices[step].append(actual_val_date)

        except Exception as e:
            # Catch errors during refitting or prediction for a specific step
            print(f"  Error during walk-forward step {i}: {e}")
            # Continue to the next iteration, potentially skipping forecasts for this point
            continue # *** Corrected indentation here ***

    print("  Walk-forward loop finished.")
    return predictions, actuals, history, prediction_indices


def calculate_validation_metrics(predictions, actuals, history, steps_to_calculate=range(1, 9)):
    """
    Calculate validation metrics (RMSE, MAE, MAPE, MSE, Directional Accuracy)
    based on the results from perform_walk_forward_validation.

    Args:
        predictions (dict): Dictionary of predictions {step: [values]}.
        actuals (dict): Dictionary of actual values {step: [values]}.
        history (dict): Dictionary of historical values {step: [values]}.
        steps_to_calculate (range or list): Steps for which to calculate metrics.

    Returns:
        dict: Dictionary where keys are 'X-Step' and values are dicts of metrics.
    """
    metrics_summary = {}
    print("  Calculating validation metrics...")

    for step in steps_to_calculate:
        preds_step = predictions.get(step, [])
        actuals_step = actuals.get(step, [])
        history_step = history.get(step, [])

        # Ensure we have results for this step and lengths match
        if len(preds_step) > 0 and len(preds_step) == len(actuals_step) and len(preds_step) == len(history_step):
            rmse = np.sqrt(mean_squared_error(actuals_step, preds_step))
            mae = mean_absolute_error(actuals_step, preds_step)
            mse = mean_squared_error(actuals_step, preds_step)
            mape = mean_absolute_percentage_error(actuals_step, preds_step)
            dacc = directional_accuracy(actuals_step, preds_step, history_step)

            metrics_summary[f'{step}-Step'] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'MSE': mse,
                'Directional Accuracy': dacc,
                'Count': len(preds_step) # Number of forecasts made for this step
            }
        else:
            # Handle cases where no predictions were made or lengths mismatch
            metrics_summary[f'{step}-Step'] = {
                'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'MSE': np.nan,
                'Directional Accuracy': np.nan, 'Count': 0
            }
            if step in predictions: # Only warn if the step exists but data is bad
                 print(f"  Warning: Could not calculate metrics for {step}-Step due to missing or mismatched data (Preds: {len(preds_step)}, Actuals: {len(actuals_step)}, Hist: {len(history_step)}).")


    return metrics_summary

# ==================================
# Reporting Helper Functions
# ==================================

def create_metrics_df(all_column_metrics):
    """
    Create a pandas DataFrame summarizing metrics across all columns and forecast steps.

    Args:
        all_column_metrics (dict): Dictionary where keys are column names and values are
                                   the metrics summary dicts from calculate_validation_metrics.

    Returns:
        pd.DataFrame: A multi-indexed DataFrame summarizing the metrics.
    """
    metrics_df_list = []
    # Iterate through each column's metrics dictionary
    for col_name, metrics in all_column_metrics.items():
        # metrics is like {'1-Step': {...}, '2-Step': {...}, ...}
        for step_key, step_metrics in metrics.items():
            # Add column and step info to each step's metrics dictionary before appending
            step_metrics['Column'] = col_name
            step_metrics['Step'] = step_key # e.g., '1-Step'
            metrics_df_list.append(step_metrics)

    if not metrics_df_list:
        print("Warning: No metrics data available to create DataFrame.")
        return pd.DataFrame() # Return empty DataFrame

    # Create DataFrame from the list of dictionaries
    metrics_df = pd.DataFrame(metrics_df_list)

    # Extract step number for proper numerical sorting
    # Use regex to find the first sequence of digits in the 'Step' string
    metrics_df['Step_Num'] = metrics_df['Step'].str.extract('(\d+)', expand=False).astype(int)

    # Define the desired order of columns
    metric_cols = ['RMSE', 'MAE', 'MAPE', 'MSE', 'Directional Accuracy', 'Count']
    # Ensure all expected metric columns exist, add NaN columns if not (robustness)
    for col in metric_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan

    # Set multi-index and select/reorder columns
    metrics_df = metrics_df.set_index(['Column', 'Step_Num'])[metric_cols]

    # Sort the DataFrame by Column name and then by Step Number
    return metrics_df.sort_index()

