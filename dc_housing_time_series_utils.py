"""
DC Housing Time Series Utilities

This module contains functions for advanced time series analysis of DC housing data,
including stationarity tests, cointegration tests, and other econometric utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.api import VAR

# Suppress warnings
warnings.filterwarnings("ignore")

def test_stationarity(series, series_name=None):
    """
    Perform ADF and KPSS tests to determine if a time series is stationary.
    
    Args:
        series (pd.Series): The time series to test
        series_name (str, optional): Name of the series for printing
        
    Returns:
        bool: True if series is determined to be stationary
    """
    # Get series name if not provided
    if series_name is None:
        series_name = getattr(series, 'name', 'Series')
    
    # Drop any missing values
    train_data = series.dropna()
    
    if len(train_data) < 10:
        print(f"Not enough data points for {series_name} to perform stationarity tests")
        return False
    
    ADF_Stationary = False
    KPSS_Stationary = False
    Stationary = False

    # Run ADFuller test
    adf_result = adfuller(train_data)
    adf_p_value = adf_result[1]

    if adf_p_value <= 0.05:
        ADF_Stationary = True
        print(f"ADF Test: Stationary (p-value: {adf_p_value:.4f})")
    else:
        print(f"ADF Test: Non-stationary (p-value: {adf_p_value:.4f})")

    # Run KPSS test
    kpss_result = kpss(train_data, regression='c')
    kpss_p_value = kpss_result[1]

    if kpss_p_value >= 0.05:
        KPSS_Stationary = True
        print(f"KPSS Test: Stationary (p-value: {kpss_p_value:.4f})")
    else:
        print(f"KPSS Test: Non-stationary on c (p-value: {kpss_p_value:.4f})\ntrying with trend regression")
        # Run KPSS test with trend regression
        kpss_result = kpss(train_data, regression='ct')
        kpss_p_value = kpss_result[1]
        if kpss_p_value >= 0.05:
            KPSS_Stationary = True
            print(f"KPSS Test: Stationary on ct(p-value: {kpss_p_value:.4f})")
        else:
            print(f"KPSS Test: Non-stationary on ct (p-value: {kpss_p_value:.4f})")
    
    # Combined interpretation of ADF and KPSS results
    if ADF_Stationary and KPSS_Stationary:
        Stationary = True
        print(f"Final Decision: Stationary")
    elif not ADF_Stationary and KPSS_Stationary:
        print(f"Final Decision: Non-stationary")
    else:
        print(f"Final Decision: Mixed results, further analysis needed")
        
    return Stationary

def create_differenced_columns(df, columns, diff_order=1, inplace=False):
    """
    Create differenced columns for specified variables.
    
    Args:
        df (pd.DataFrame): DataFrame containing the columns to difference
        columns (list): List of column names to difference
        diff_order (int): Differencing order (1 or 2)
        inplace (bool): Whether to modify df in place
        
    Returns:
        pd.DataFrame: DataFrame with added differenced columns
    """
    if not inplace:
        df = df.copy()
    
    for col_name in columns:
        # First difference
        diff_col = f'{col_name}_diff1'
        df[diff_col] = df[col_name].diff()
        
        # Second difference if requested
        if diff_order == 2:
            diff_col2 = f'{col_name}_diff2'
            df[diff_col2] = df[diff_col].diff()
    
    return df

def plot_seasonal_decomposition(df, var_name, period=4):
    """
    Plot seasonal decomposition of a time series.
    
    Args:
        df (pd.DataFrame): DataFrame containing the series
        var_name (str): Name of the column to decompose
        period (int): Periodicity of the seasonal component
    """
    series = df[var_name]
    stl = STL(series, period=period)
    result = stl.fit()
    
    plt.figure(figsize=(14, 14))
    result.plot()
    plt.suptitle(f'Seasonal Decomposition of {var_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.001, 1, 1])
    plt.show()

def run_johansen_test(df, vars_i1, vars_i2=None, max_lags=8, det_order=0):
    """
    Perform Johansen cointegration test on the variables.
    
    Args:
        df (pd.DataFrame): DataFrame containing the variables
        vars_i1 (list): List of I(1) variable names
        vars_i2 (list, optional): List of I(2) variable names
        max_lags (int): Maximum lag order to test
        det_order (int): Deterministic term specification (0=constant in CE)
        
    Returns:
        tuple: (selected lag order, trace test results, max eigenvalue test results)
    """
    # Copy only the needed variables
    required_cols = vars_i1.copy()
    if vars_i2:
        required_cols.extend(vars_i2)
    
    # Check if all specified columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: The following required columns are missing: {missing_cols}")

    # Create the first difference of I(2) variables if any
    df_test = df[vars_i1].copy()
    if vars_i2:
        for var in vars_i2:
            diff_col = f'{var}_diff1'
            if diff_col not in df.columns:
                df[diff_col] = df[var].diff()
            df_test[f'{var}_diff1'] = df[diff_col]
    
    # Handle missing values
    df_test_dropna = df_test.dropna()
    
    # Select the optimal lag order
    lag_order_results = vecm.select_order(
        data=df_test_dropna, 
        maxlags=max_lags, 
        deterministic='co'  # Corresponds to det_order=0
    )
    
    # Get the BIC optimal lag order
    k_ar_diff_selected = lag_order_results.bic
    
    # Perform Johansen test
    result = vecm.coint_johansen(
        df_test_dropna, 
        det_order=det_order, 
        k_ar_diff=k_ar_diff_selected
    )
    
    return k_ar_diff_selected, result

def display_johansen_results(result, alpha=0.05):
    """
    Display and interpret Johansen test results.
    
    Args:
        result: Johansen test result object
        alpha (float): Significance level (0.05 = 95% confidence)
        
    Returns:
        int: The cointegrating rank determined by the test
    """
    # Find column index for critical values at given alpha
    crit_col = 1  # 95% by default (0=90%, 1=95%, 2=99%)
    
    print("\n--- Johansen Test Results ---")
    
    print("\nTrace Statistic Test:")
    print("r | trace_stat | 90% crit | 95% crit | 99% crit")
    print("--|------------|----------|----------|----------")
    for i in range(len(result.lr1)):
        print(f"{i:<2}| {result.lr1[i]:<10.2f} | {result.cvt[i, 0]:<8.2f} | {result.cvt[i, 1]:<8.2f} | {result.cvt[i, 2]:<8.2f}")
    
    print("\nMax Eigenvalue Statistic Test:")
    print("r | max_eig_stat | 90% crit | 95% crit | 99% crit")
    print("--|--------------|----------|----------|----------")
    for i in range(len(result.lr2)):
        print(f"{i:<2}| {result.lr2[i]:<12.2f} | {result.cvm[i, 0]:<8.2f} | {result.cvm[i, 1]:<8.2f} | {result.cvm[i, 2]:<8.2f}")
    
    # Determine rank based on trace test
    rank_trace = 0
    for i in range(len(result.lr1)):
        if result.lr1[i] > result.cvt[i, crit_col]:
            rank_trace += 1
        else:
            break
    print(f"\n- Trace test indicates cointegrating rank (r) <= {rank_trace}")
    
    # Determine rank based on max eigenvalue test
    rank_max_eig = 0
    for i in range(len(result.lr2)):
        if result.lr2[i] > result.cvm[i, crit_col]:
            rank_max_eig += 1
        else:
            break
    print(f"- Max Eigenvalue test indicates cointegrating rank (r) = {rank_max_eig}")
    
    # Conclusion
    final_rank = -1
    print("\n--- Conclusion ---")
    if rank_trace == 0 and rank_max_eig == 0:
        print("Both tests suggest No Cointegration (rank r=0).")
        print("Recommendation: Use a VAR model on the appropriately differenced (stationary) data.")
        final_rank = 0
    elif rank_trace > 0 and rank_max_eig > 0:
        final_rank = rank_max_eig
        if rank_trace < rank_max_eig:
            print(f"Warning: Trace test suggests lower rank ({rank_trace}) than Max-Eigenvalue test ({rank_max_eig}). Using r={final_rank} based on Max-Eigenvalue.")
        elif rank_trace > rank_max_eig:
            print(f"Trace test suggests higher rank ({rank_trace}) than Max-Eigenvalue test ({rank_max_eig}). Using r={final_rank} based on Max-Eigenvalue.")
        else:
            print(f"Both tests suggest cointegrating rank r = {final_rank}.")
            
        if final_rank > 0:
            print(f"Recommendation: Cointegration detected (rank r={final_rank}). A VECM is appropriate.")
    elif rank_trace == 0 and rank_max_eig > 0:
        print(f"Mixed Results: Trace test suggests r=0, Max-Eigenvalue test suggests r={rank_max_eig}.")
        print("Recommendation: Investigate further. Check sensitivity to lag order, deterministic terms.")
    elif rank_trace > 0 and rank_max_eig == 0:
        print(f"Mixed Results: Trace test suggests r={rank_trace}, Max-Eigenvalue test suggests r=0.")
        print("Recommendation: Investigate further. Check sensitivity to lag order, deterministic terms.")
    
    return final_rank

def fit_vecm_model(df, variables, cointegration_rank, lag_order=1, deterministic_term='co', exog=None):
    """
    Fit a Vector Error Correction Model (VECM).
    
    Args:
        df (pd.DataFrame): DataFrame containing the variables
        variables (list): List of variable names to include in the model
        cointegration_rank (int): Cointegrating rank from Johansen test
        lag_order (int): Lag order for the VECM (k_ar_diff)
        deterministic_term (str): Deterministic terms specification ('co', 'ci', 'nc')
        exog (pd.DataFrame, optional): Exogenous variables
        
    Returns:
        vecm results object
    """
    # Prepare endogenous variables
    df_vecm = df[variables].copy()
    
    # Handle missing values
    df_vecm_dropna = df_vecm.dropna()
    
    # Instantiate and fit the model
    model = vecm.VECM(
        endog=df_vecm_dropna,
        exog=exog,
        k_ar_diff=lag_order,
        coint_rank=cointegration_rank,
        deterministic=deterministic_term
    )
    
    # Fit the model
    vecm_results = model.fit()
    
    return vecm_results

def plot_residuals(results, title=None):
    """
    Plot residuals from a fitted VAR or VECM model.
    
    Args:
        results: VAR or VECM results object
        title (str, optional): Title for the plot
    """
    # Get the residuals
    residuals = results.resid
    
    # Create DataFrame for plotting
    variables = results.names if hasattr(results, 'names') else [f'y{i+1}' for i in range(residuals.shape[1])]
    residuals_df = pd.DataFrame(
        residuals,
        index=results.model.data.dates[results.k_ar:] if hasattr(results.model, 'data') and hasattr(results.model.data, 'dates') else None,
        columns=[f'{var}_resid' for var in variables]
    )
    
    # Determine layout
    num_variables = residuals_df.shape[1]
    num_cols = min(4, num_variables)
    num_rows = int(np.ceil(num_variables / num_cols))
    
    # Create plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3), sharex=True)
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    
    for i, col in enumerate(residuals_df.columns):
        if i < len(axes):
            residuals_df[col].plot(ax=axes[i], title=col)
            axes[i].set_ylabel('Residual Value')
            axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plot_title = title or 'Model Residuals Over Time'
    fig.suptitle(plot_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def fit_var_model(df, variables, lag_order=None, max_lags=8, trend='ct', exog=None):
    """
    Fit a Vector Autoregression (VAR) model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the variables
        variables (list): List of variable names to include
        lag_order (int, optional): Specific lag order to use, if None will select automatically
        max_lags (int): Maximum lags to consider when selecting optimal lag order
        trend (str): Trend specification ('c', 'ct', 'nc')
        exog (pd.DataFrame, optional): Exogenous variables
        
    Returns:
        tuple: (VAR model, VAR results object, selected lag order)
    """
    # Prepare data
    df_var = df[variables].copy()
    df_var_dropna = df_var.dropna()
    
    # Instantiate VAR model
    model = VAR(endog=df_var_dropna, exog=exog)
    
    # Select lag order if not specified
    if lag_order is None:
        lag_order_results = model.select_order(maxlags=max_lags)
        print("\n--- Lag Selection Results ---")
        print(lag_order_results.summary())
        
        # Use AIC criteria for lag selection
        lag_order = lag_order_results.aic
        print(f"\nSelected lag order (AIC): {lag_order}")
    
    # Fit the model
    results = model.fit(maxlags=lag_order, trend=trend)
    
    return model, results, lag_order

def create_dummies(df, periods, inplace=False):
    """
    Create dummy variables for specific time periods.
    
    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex
        periods (dict): Dictionary mapping dummy names to (start_date, end_date) tuples
        inplace (bool): Whether to modify df in place
        
    Returns:
        pd.DataFrame: DataFrame with added dummy variables
    """
    if not inplace:
        df = df.copy()
    
    for dummy_name, (start_date, end_date) in periods.items():
        # Initialize column with zeros
        df[dummy_name] = 0
        
        # Set values to 1 for dates between start and end (inclusive)
        df.loc[(df.index >= start_date) & (df.index <= end_date), dummy_name] = 1
        print(f"Created '{dummy_name}' (active {start_date} to {end_date})")
    
    return df

def create_pct_change_columns(df, exclude_cols=None, inplace=False):
    """
    Create percentage change columns for all variables except those in exclude_cols.
    
    Args:
        df (pd.DataFrame): DataFrame containing the variables
        exclude_cols (list, optional): List of column names to exclude
        inplace (bool): Whether to modify df in place
        
    Returns:
        pd.DataFrame: DataFrame with added percentage change columns
    """
    if not inplace:
        df = df.copy()
    
    exclude_cols = exclude_cols or []
    
    for col in df.columns:
        if col not in exclude_cols and not col.endswith('_pct_change'):
            df[f'{col}_pct_change'] = df[col].pct_change()
    
    return df