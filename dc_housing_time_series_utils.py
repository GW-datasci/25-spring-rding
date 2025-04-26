"""
DC Housing Time Series Utilities

This module contains functions for advanced time series analysis of DC housing data,
including stationarity tests, cointegration tests, VECM/VAR modeling, and other utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECMResults
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

# Suppress warnings
warnings.filterwarnings("ignore")

def test_stationarity(series, series_name=None):
    """
    Perform ADF and KPSS tests to determine if a time series is stationary.

    Args:
        series (pd.Series): The time series to test. Drops NaNs internally.
        series_name (str, optional): Name of the series for printing. Defaults to series.name.

    Returns:
        str: 'Stationary', 'Non-stationary', or 'Mixed results'.
    """
    # Get series name if not provided
    if series_name is None:
        series_name = getattr(series, 'name', 'Series')

    # Drop any missing values ONLY for the test
    test_data = series.dropna()

    if len(test_data) < 10:
        print(f"Not enough data points for {series_name} to perform stationarity tests (need at least 10, got {len(test_data)})")
        return 'Insufficient data'

    adf_stationary = False
    kpss_stationary = False
    final_decision = "Mixed results, further analysis needed" # Default

    # --- ADF Test ---
    try:
        adf_result = adfuller(test_data)
        adf_p_value = adf_result[1]
        if adf_p_value <= 0.05:
            adf_stationary = True
            print(f"ADF Test: Stationary (p-value: {adf_p_value:.4f})")
        else:
            print(f"ADF Test: Non-stationary (p-value: {adf_p_value:.4f})")
    except Exception as e:
        print(f"ADF Test failed for {series_name}: {e}")
        adf_p_value = np.nan # Indicate failure

    # --- KPSS Test ---
    try:
        # Test with constant
        kpss_result_c = kpss(test_data, regression='c', nlags="auto")
        kpss_p_value_c = kpss_result_c[1]

        if kpss_p_value_c >= 0.05:
            kpss_stationary = True
            print(f"KPSS Test: Stationary (level, p-value: {kpss_p_value_c:.4f})")
        else:
            print(f"KPSS Test: Non-stationary (level, p-value: {kpss_p_value_c:.4f}) -> trying with trend...")
            # Test with constant and trend if failed with constant
            try:
                kpss_result_ct = kpss(test_data, regression='ct', nlags="auto")
                kpss_p_value_ct = kpss_result_ct[1]
                if kpss_p_value_ct >= 0.05:
                    kpss_stationary = True # Considered stationary if stationary around trend
                    print(f"KPSS Test: Stationary (trend, p-value: {kpss_p_value_ct:.4f})")
                else:
                    print(f"KPSS Test: Non-stationary (trend, p-value: {kpss_p_value_ct:.4f})")
            except Exception as e_ct:
                 print(f"KPSS Test (trend) failed for {series_name}: {e_ct}")
                 kpss_p_value_ct = np.nan # Indicate failure

    except Exception as e:
        print(f"KPSS Test (level) failed for {series_name}: {e}")
        kpss_p_value_c = np.nan # Indicate failure


    # --- Final Decision Logic ---
    if adf_stationary and kpss_stationary:
        final_decision = "Stationary"
    elif not adf_stationary and not kpss_stationary:
         # Both agree it's non-stationary (KPSS failed level and maybe trend)
         final_decision = "Non-stationary"
    elif not adf_stationary and kpss_stationary:
         # Difference stationary: ADF says non-stationary, KPSS says stationary (around level or trend)
         # This often implies differencing is needed.
         final_decision = "Non-stationary (likely difference stationary)"
    elif adf_stationary and not kpss_stationary:
         # Trend stationary: ADF says stationary (rejects unit root), KPSS says non-stationary (rejects level/trend stationarity)
         # This often implies detrending is needed, or the series is stationary but with structure KPSS picks up.
         # For practical VAR/VECM, often treated as needing differencing if KPSS rejects level stationarity.
         final_decision = "Mixed results (potentially trend stationary, check KPSS type)"


    print(f"Final Decision: {final_decision}")
    return final_decision


def create_differenced_columns(df, columns, diff_order=1, inplace=False):
    """
    Create differenced columns for specified variables.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to difference.
        columns (list): List of column names to difference.
        diff_order (int): Differencing order (default is 1). Creates _diff1, _diff2 etc.
        inplace (bool): Whether to modify df in place.

    Returns:
        pd.DataFrame: DataFrame with added differenced columns (or None if inplace=True).
    """
    df_out = df if inplace else df.copy()

    for col_name in columns:
        if col_name not in df_out.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping differencing.")
            continue
        current_series = df_out[col_name]
        for i in range(1, diff_order + 1):
            # Construct the name for the i-th difference
            # If diff_order=1, name is col_name_diff1
            # If diff_order=2, names are col_name_diff1, col_name_diff2
            diff_col_name = f'{col_name}_diff{i}'
            current_series = current_series.diff()
            df_out[diff_col_name] = current_series
            print(f"Created '{diff_col_name}'")

    if not inplace:
        return df_out
    return None

def plot_seasonal_decomposition(df, var_name, period=4):
    """
    Plot seasonal decomposition of a time series using STL.

    Args:
        df (pd.DataFrame): DataFrame containing the series.
        var_name (str): Name of the column to decompose.
        period (int): Periodicity of the seasonal component (e.g., 4 for quarterly).
    """
    if var_name not in df.columns:
        print(f"Error: Column '{var_name}' not found for seasonal decomposition.")
        return

    series = df[var_name].dropna() # Drop NaNs before decomposition

    if len(series) < 2 * period:
         print(f"Warning: Series '{var_name}' has length {len(series)}, which is less than 2 * period ({2*period}). STL decomposition may be unreliable.")
         # Optionally return or proceed with caution
         # return

    try:
        stl = STL(series, period=period, robust=True) # Use robust=True for potential outliers
        result = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True) # Create 4 subplots

        axes[0].plot(result.observed)
        axes[0].set_ylabel('Observed')

        axes[1].plot(result.trend)
        axes[1].set_ylabel('Trend')

        axes[2].plot(result.seasonal)
        axes[2].set_ylabel('Seasonal')

        axes[3].plot(result.resid)
        axes[3].set_ylabel('Residual')
        axes[3].axhline(0, linestyle='--', color='grey') # Add horizontal line at 0 for residuals

        fig.suptitle(f'Seasonal Decomposition of {var_name} (STL)', fontsize=16)
        plt.xlabel("Date")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    except Exception as e:
        print(f"Error during STL decomposition for '{var_name}': {e}")


def run_johansen_test(df, vars_i1, vars_i2=None, max_lags=8, det_order=0):
    """
    Perform Johansen cointegration test on the variables. Handles I(1) and I(2) variables.

    Args:
        df (pd.DataFrame): DataFrame containing the variables (levels).
        vars_i1 (list): List of I(1) variable names (use levels).
        vars_i2 (list, optional): List of I(2) variable names (use levels).
        max_lags (int): Maximum lag order for underlying VAR.
        det_order (int): Deterministic term specification for Johansen test.
                         -1: No deterministic terms.
                          0: Constant term in cointegrating relation (default).
                          1: Linear trend in cointegrating relation.

    Returns:
        tuple: (selected lag order k_ar_diff, JohansenResults object) or (None, None) if error.
    """
    vars_i2 = vars_i2 or [] # Ensure vars_i2 is a list

    # --- Data Preparation ---
    # Select necessary columns (levels)
    required_cols = vars_i1 + vars_i2
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns for Johansen test are missing: {missing_cols}")
        return None, None

    df_levels = df[required_cols].copy()

    # Create the first difference of I(2) variables
    df_test_input = df_levels[vars_i1].copy() # Start with I(1) levels
    for var_i2 in vars_i2:
        diff_col_name = f'{var_i2}_diff1'
        df_test_input[diff_col_name] = df_levels[var_i2].diff() # Add first diff of I(2) vars

    # Drop rows with NaN values created by differencing I(2) vars
    df_test_dropna = df_test_input.dropna()

    if len(df_test_dropna) < 2 * len(required_cols): # Basic check for sufficient data
         print(f"Warning: Not enough observations ({len(df_test_dropna)}) after differencing for Johansen test.")
         # return None, None # Or proceed cautiously

    # --- Determine Deterministic Term String for select_order ---
    # select_order uses string codes slightly differently than coint_johansen
    if det_order == -1:
        det_string = 'nc' # No constant
    elif det_order == 0:
        det_string = 'co' # Constant in cointegrating equation only
    elif det_order == 1:
        det_string = 'ci' # Constant outside cointegrating equation (intercept in VECM)
    # Note: 'lo' (linear trend outside) and 'li' (linear trend inside) also exist
    else:
        print(f"Warning: Unsupported det_order {det_order} for select_order. Using 'co'.")
        det_string = 'co'


    # --- Lag Order Selection for the underlying VAR ---
    try:
        print(f"Selecting lag order for VAR underlying Johansen test (max_lags={max_lags}, det_term='{det_string}')...")
        # Use the data prepared for the test (I(1) levels, I(2) first differences)
        lag_order_results = vecm.select_order(
            data=df_test_dropna,
            maxlags=max_lags,
            deterministic=det_string,
            seasons=0 # Assuming non-seasonal for now
        )
        # Select lag based on BIC (often preferred for consistency)
        k_ar_diff_selected = lag_order_results.bic
        print(lag_order_results) # Show all criteria results
        print(f"Selected lag order k_ar_diff = {k_ar_diff_selected} (using BIC)")

    except Exception as e:
        print(f"Error during lag order selection for Johansen test: {e}")
        return None, None

    # --- Perform Johansen Test ---
    try:
        print(f"Performing Johansen test with k_ar_diff={k_ar_diff_selected} and det_order={det_order}...")
        # Use the same prepared data
        johansen_result = vecm.coint_johansen(
            endog=df_test_dropna,
            det_order=det_order,
            k_ar_diff=k_ar_diff_selected # Lags in the VAR representation
        )
        return k_ar_diff_selected, johansen_result

    except Exception as e:
        print(f"Error during Johansen cointegration test: {e}")
        return k_ar_diff_selected, None # Return lag order even if test fails


def display_johansen_results(result, alpha=0.05):
    """
    Display and interpret Johansen test results nicely.

    Args:
        result (JohansenResults): The result object from vecm.coint_johansen.
        alpha (float): Significance level (0.10, 0.05, 0.01).

    Returns:
        int: The cointegrating rank suggested by the Max Eigenvalue test (common choice).
             Returns -1 if results are inconclusive or tests disagree significantly.
    """
    if result is None:
        print("No Johansen results to display.")
        return -1

    # Determine critical value column index based on alpha
    if alpha == 0.10:
        crit_col_idx = 0
        crit_level = "90%"
    elif alpha == 0.05:
        crit_col_idx = 1
        crit_level = "95%"
    elif alpha == 0.01:
        crit_col_idx = 2
        crit_level = "99%"
    else:
        print(f"Warning: Invalid alpha {alpha}. Using alpha=0.05 (95%).")
        crit_col_idx = 1
        crit_level = "95%"

    num_vars = result.eig.shape[0] # Number of variables in the test

    print("\n--- Johansen Test Results ---")
    print(f"Significance Level: {alpha*100:.0f}%")

    # --- Trace Statistic Test ---
    print("\nTrace Statistic Test (H0: rank <= r):")
    print(f"{'r':<3} | {'trace_stat':<12} | {crit_level+' crit':<10} | {'Decision':<15}")
    print("-" * (3 + 1 + 12 + 3 + 10 + 3 + 15))
    rank_trace = 0
    for i in range(num_vars):
        hypothesized_rank = i
        trace_stat = result.lr1[i]
        crit_val = result.cvt[i, crit_col_idx]
        reject_null = trace_stat > crit_val
        decision = "Reject H0" if reject_null else "Fail to Reject H0"
        print(f"{hypothesized_rank:<3} | {trace_stat:<12.3f} | {crit_val:<10.3f} | {decision:<15}")
        if reject_null:
            rank_trace = i + 1 # If we reject r<=i, then rank is at least i+1
        else:
            break # Stop at the first failure to reject
    print(f"Trace test suggests cointegrating rank (r) = {rank_trace}")

    # --- Max Eigenvalue Statistic Test ---
    print("\nMax Eigenvalue Statistic Test (H0: rank = r vs H1: rank = r+1):")
    print(f"{'r':<3} | {'max_eig_stat':<12} | {crit_level+' crit':<10} | {'Decision':<15}")
    print("-" * (3 + 1 + 12 + 3 + 10 + 3 + 15))
    rank_max_eig = 0
    for i in range(num_vars):
        hypothesized_rank = i
        max_eig_stat = result.lr2[i]
        crit_val = result.cvm[i, crit_col_idx]
        reject_null = max_eig_stat > crit_val
        decision = "Reject H0" if reject_null else "Fail to Reject H0"
        print(f"{hypothesized_rank:<3} | {max_eig_stat:<12.3f} | {crit_val:<10.3f} | {decision:<15}")
        if reject_null:
            rank_max_eig = i + 1 # If we reject r=i, then rank is at least i+1
        else:
            break # Stop at the first failure to reject
    print(f"Max Eigenvalue test suggests cointegrating rank (r) = {rank_max_eig}")

    # --- Conclusion ---
    final_rank = -1 # Default to inconclusive
    print("\n--- Conclusion ---")
    if rank_trace == rank_max_eig:
        final_rank = rank_trace
        if final_rank == 0:
            print(f"Both tests suggest No Cointegration (rank r=0).")
            print("Recommendation: Use VAR on appropriately differenced data.")
        else:
            print(f"Both tests agree: Cointegrating Rank r = {final_rank}.")
            print(f"Recommendation: Cointegration detected. A VECM with rank {final_rank} is appropriate.")
    else:
        # Tests disagree
        print(f"Tests Disagree: Trace suggests r={rank_trace}, Max-Eigenvalue suggests r={rank_max_eig}.")
        # Common practice: often rely on Max-Eigenvalue, but caution is needed.
        final_rank = rank_max_eig # Tentatively use Max-Eigenvalue
        print(f"Recommendation: Tests conflict. Cautiously proceed with r={final_rank} (from Max-Eigenvalue),")
        print("                 but investigate sensitivity to lag order, deterministic terms, or consider alternative methods.")
        # final_rank = -1 # Or mark as inconclusive

    return final_rank


def fit_vecm_model(df, variables, cointegration_rank, lag_order=1, deterministic_term='co', exog=None):
    """
    Fit a Vector Error Correction Model (VECM). Assumes variables are I(1) or prepared correctly.

    Args:
        df (pd.DataFrame): DataFrame containing the variables (levels or prepared I(1)/diff(I(2))).
                           Should align with the data used for Johansen test.
        variables (list): List of endogenous variable names to include in the model.
        cointegration_rank (int): Cointegrating rank from Johansen test.
        lag_order (int): Lag order for the VECM (k_ar_diff from Johansen/VAR selection).
        deterministic_term (str): Deterministic terms specification ('nc', 'co', 'ci', 'lo', 'li').
                                  'co' = constant in CE. 'ci' = constant in VECM.
        exog (pd.DataFrame, optional): DataFrame of exogenous variables, aligned with df.

    Returns:
        VECMResults object or None if fitting fails.
    """
    # Prepare endogenous variables
    df_vecm_endog = df[variables].copy()

    # Align exogenous variables if provided
    df_exog_aligned = None
    if exog is not None:
        if not isinstance(exog, pd.DataFrame):
             print("Error: Exogenous data must be a pandas DataFrame.")
             return None
        # Align index with endogenous data BEFORE dropping NaNs
        df_exog_aligned = exog.reindex(df_vecm_endog.index)

    # Combine endog and aligned exog (if any) to handle NaNs consistently
    if df_exog_aligned is not None:
        combined_df = pd.concat([df_vecm_endog, df_exog_aligned], axis=1)
    else:
        combined_df = df_vecm_endog

    # Drop rows with any NaNs across endogenous or exogenous variables
    combined_df_dropna = combined_df.dropna()

    # Separate back into endogenous and exogenous after dropping NaNs
    df_vecm_final_endog = combined_df_dropna[variables]
    df_exog_final = combined_df_dropna[exog.columns] if exog is not None else None

    print(f"Final endogenous data shape for VECM: {df_vecm_final_endog.shape}")
    if df_exog_final is not None:
        print(f"Final exogenous data shape for VECM: {df_exog_final.shape}")
    if len(df_vecm_final_endog) == 0:
        print("Error: All data dropped due to NaNs. Cannot fit VECM.")
        return None

    # Instantiate and fit the model
    try:
        model = vecm.VECM(
            endog=df_vecm_final_endog,
            exog=df_exog_final,
            k_ar_diff=lag_order,         # Lags in VAR representation
            coint_rank=cointegration_rank,
            deterministic=deterministic_term
        )

        vecm_results = model.fit()
        print(f"VECM fitted successfully.")
        return vecm_results

    except Exception as e:
        print(f"Error fitting VECM model: {e}")
        # You might want to print more details about the shapes and data here for debugging
        # print("Endog data head:\n", df_vecm_final_endog.head())
        # if df_exog_final is not None:
        #     print("Exog data head:\n", df_exog_final.head())
        return None


def plot_residuals(results, title=None):
    """
    Plot residuals from a fitted VAR or VECM model.

    Args:
        results: VARResultsWrapper or VECMResults object.
        title (str, optional): Title for the plot.
    """
    if results is None:
        print("No model results provided for plotting residuals.")
        return

    try:
        # --- Get Residuals ---
        residuals = results.resid
        if residuals is None or residuals.shape[0] == 0:
            print("Error: Residuals are not available or empty.")
            return

        num_residuals, num_variables = residuals.shape
        model_nobs = results.nobs # Number of observations used in estimation
        k_ar = results.k_ar # Number of lags in the model

        # --- Get Variable Names ---
        variables = results.names if hasattr(results, 'names') else [f'resid_{i+1}' for i in range(num_variables)]

        # --- Get Index (Dates) ---
        residual_index = None

        # Try to get the index from the original data used for fitting
        # For VARResultsWrapper: results.model.y contains the data used
        # For VECMResults: results.model.endog contains the data used
        data_used_for_fitting = None
        if isinstance(results, VARResultsWrapper) and hasattr(results.model, 'y'):
            data_used_for_fitting = results.model.y # This is usually a numpy array
            # Get index from the original DataFrame passed to VAR if possible
            if hasattr(results.model, 'endog_names') and hasattr(results.model, 'data') and hasattr(results.model.data, 'orig_endog'):
                 original_df_index = results.model.data.orig_endog.index
                 if len(original_df_index) >= k_ar + model_nobs:
                     residual_index = original_df_index[k_ar : k_ar + model_nobs]

        elif isinstance(results, VECMResults) and hasattr(results.model, 'endog'):
             data_used_for_fitting = results.model.endog # This should be the DataFrame used
             if isinstance(data_used_for_fitting, pd.DataFrame):
                  residual_index = data_used_for_fitting.index # Index should align directly

        # Fallback if index extraction failed or lengths mismatch
        if residual_index is None or len(residual_index) != num_residuals:
            print("Warning: Could not accurately determine residual dates from model attributes. Using RangeIndex.")
            residual_index = pd.RangeIndex(start=k_ar, stop=k_ar + num_residuals, step=1)


        # --- Create DataFrame for Plotting ---
        # Ensure residuals are numpy array before creating DataFrame if needed
        residuals_array = residuals if isinstance(residuals, np.ndarray) else residuals.values

        residuals_df = pd.DataFrame(
            residuals_array,
            index=residual_index,
            columns=[f'{var}_resid' for var in variables]
        )

        # --- Plotting ---
        if num_residuals == 0:
             print("No residuals to plot.")
             return

        num_cols = min(3, num_variables) # Adjust number of columns
        num_rows = int(np.ceil(num_variables / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(max(15, num_cols * 5), num_rows * 3.5), sharex=True, squeeze=False)
        axes_flat = axes.flatten()

        for i, col in enumerate(residuals_df.columns):
            if i < len(axes_flat):
                try:
                    residuals_df[col].plot(ax=axes_flat[i], legend=False)
                    axes_flat[i].set_title(col, fontsize=11)
                    axes_flat[i].set_ylabel('Residual', fontsize=9)
                    axes_flat[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
                    axes_flat[i].tick_params(axis='x', labelsize=9)
                    axes_flat[i].tick_params(axis='y', labelsize=9)
                    # Add grid for easier reading
                    axes_flat[i].grid(True, linestyle='--', alpha=0.6)
                except Exception as plot_err:
                     print(f"Error plotting column {col}: {plot_err}")
                     axes_flat[i].set_title(f"{col}\n(Plotting Error)", fontsize=10)


        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plot_title = title or 'Model Residuals Over Time'
        fig.suptitle(plot_title, fontsize=16, y=0.99) # Adjust title position
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plt.show()

    except AttributeError as ae:
         print(f"Error plotting residuals: Attribute missing - {ae}. Check the results object structure.")
    except Exception as e:
        print(f"An unexpected error occurred during residual plotting: {e}")


def fit_var_model(df, variables, lag_order=None, max_lags=8, trend='ct', exog=None):
    """
    Fit a Vector Autoregression (VAR) model. Assumes variables are stationary.

    Args:
        df (pd.DataFrame): DataFrame containing the stationary variables.
        variables (list): List of endogenous variable names to include.
        lag_order (int, optional): Specific lag order. If None, selects using AIC.
        max_lags (int): Maximum lags for automatic selection.
        trend (str): Trend specification ('nc', 'c', 'ct', 'ctt').
        exog (pd.DataFrame, optional): DataFrame of exogenous variables, aligned with df.

    Returns:
        tuple: (VAR model object, VARResultsWrapper object, selected lag order) or (None, None, None) if error.
    """
    # Prepare endogenous data
    df_var_endog = df[variables].copy()

     # Align exogenous variables if provided
    df_exog_aligned = None
    if exog is not None:
        if not isinstance(exog, pd.DataFrame):
             print("Error: Exogenous data must be a pandas DataFrame.")
             return None, None, None
        # Align index with endogenous data BEFORE dropping NaNs
        df_exog_aligned = exog.reindex(df_var_endog.index)

    # Combine endog and aligned exog (if any) to handle NaNs consistently
    if df_exog_aligned is not None:
        combined_df = pd.concat([df_var_endog, df_exog_aligned], axis=1)
    else:
        combined_df = df_var_endog

    # Drop rows with any NaNs across endogenous or exogenous variables
    combined_df_dropna = combined_df.dropna()

    # Separate back into endogenous and exogenous after dropping NaNs
    df_var_final_endog = combined_df_dropna[variables]
    df_exog_final = combined_df_dropna[exog.columns] if exog is not None else None

    print(f"Final endogenous data shape for VAR: {df_var_final_endog.shape}")
    if df_exog_final is not None:
        print(f"Final exogenous data shape for VAR: {df_exog_final.shape}")

    if len(df_var_final_endog) == 0:
        print("Error: All data dropped due to NaNs. Cannot fit VAR.")
        return None, None, None


    # Instantiate VAR model
    try:
        model = VAR(endog=df_var_final_endog, exog=df_exog_final)

        # Select lag order if not specified
        selected_lag = lag_order
        if selected_lag is None:
            print(f"\nSelecting VAR lag order (max_lags={max_lags}, trend='{trend}')...")
            # Note: select_order doesn't use exog, fit handles it.
            lag_order_results = model.select_order(maxlags=max_lags, trend=trend)
            print("\n--- Lag Selection Results ---")
            print(lag_order_results.summary())
            selected_lag = lag_order_results.aic # Use AIC by default
            print(f"\nSelected lag order (AIC): {selected_lag}")
            if selected_lag == 0:
                print("Warning: Selected lag order is 0. VAR may not capture dynamics.")
                # You might want to force a minimum lag, e.g., selected_lag = 1

        # Fit the model
        print(f"\nFitting VAR model with lag order = {selected_lag} and trend = '{trend}'...")
        results = model.fit(maxlags=selected_lag, trend=trend)
        print("VAR model fitted successfully.")
        return model, results, selected_lag

    except ValueError as ve:
         print(f"Error fitting VAR model: {ve}")
         print("This might be due to insufficient data after differencing/NaN handling for the chosen lags.")
         return None, None, lag_order # Return selected lag even if fit fails
    except Exception as e:
        print(f"An unexpected error occurred during VAR fitting: {e}")
        return None, None, lag_order


def create_dummies(df, periods, inplace=False):
    """
    Create dummy variables (0 or 1) for specific time periods in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        periods (dict): Dictionary mapping dummy variable names (str) to
                        (start_date_str, end_date_str) tuples. Dates are inclusive.
        inplace (bool): Whether to modify the DataFrame in place.

    Returns:
        pd.DataFrame: DataFrame with added dummy variables (or None if inplace=True).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index must be a DatetimeIndex to create time-based dummies.")
        return df if not inplace else None

    df_out = df if inplace else df.copy()

    for dummy_name, (start_date_str, end_date_str) in periods.items():
        try:
            # Convert string dates to datetime objects for comparison
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)

            # Create the dummy column, initialized to 0
            df_out[dummy_name] = 0

            # Set values to 1 for dates within the specified range (inclusive)
            df_out.loc[(df_out.index >= start_date) & (df_out.index <= end_date), dummy_name] = 1
            print(f"Created '{dummy_name}' (active {start_date_str} to {end_date_str})")

        except Exception as e:
            print(f"Error creating dummy '{dummy_name}' for period {start_date_str}-{end_date_str}: {e}")

    if not inplace:
        return df_out
    return None


def create_pct_change_columns(df, exclude_cols=None, inplace=False):
    """
    Create percentage change columns for numeric variables in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the variables.
        exclude_cols (list, optional): List of column names to exclude from pct_change calculation.
        inplace (bool): Whether to modify the DataFrame in place.

    Returns:
        pd.DataFrame: DataFrame with added percentage change columns (or None if inplace=True).
    """
    df_out = df if inplace else df.copy()
    exclude_cols = exclude_cols or []

    for col in df.columns:
        # Check if column should be excluded or is already a pct_change column
        if col in exclude_cols or col.endswith('_pct_change'):
            continue

        # Check if the column is numeric before calculating pct_change
        if pd.api.types.is_numeric_dtype(df_out[col]):
            pct_change_col_name = f'{col}_pct_change'
            df_out[pct_change_col_name] = df_out[col].pct_change()
            # Optional: print(f"Created '{pct_change_col_name}'")
        else:
            print(f"Skipping non-numeric column '{col}' for percentage change calculation.")

    if not inplace:
        return df_out
    return None