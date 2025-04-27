# dc_housing_lstm_utils.py (v3 - Fix iloc error)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import time
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set seeds for reproducibility (optional, but good practice)
# tf.random.set_seed(42)
# np.random.seed(42)

# --- Helper Metric Functions (from original script) ---
def mean_absolute_percentage_error_np(y_true, y_pred):
    """Numpy implementation of MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.nan
    # Ensure y_true[mask] is not empty before division
    if len(y_true[mask]) == 0: return np.nan
    # Ensure y_pred corresponding to the mask is used
    if len(y_pred[mask]) == 0: return np.nan
    # Ensure lengths match after masking
    if len(y_true[mask]) != len(y_pred[mask]):
        print(f"Warning: MAPE length mismatch after masking. y_true_masked={len(y_true[mask])}, y_pred_masked={len(y_pred[mask])}")
        # Attempt to align based on minimum length if caused by edge cases
        min_len = min(len(y_true[mask]), len(y_pred[mask]))
        if min_len == 0: return np.nan
        return np.mean(np.abs((y_true[mask][:min_len] - y_pred[mask][:min_len]) / y_true[mask][:min_len])) * 100

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy_np(y_true, y_pred, y_true_prev):
    """Numpy implementation of Directional Accuracy"""
    y_true, y_pred, y_true_prev = np.array(y_true), np.array(y_pred), np.array(y_true_prev)

    # Ensure all arrays have the same length
    min_len = min(len(y_true), len(y_pred), len(y_true_prev))
    if min_len == 0:
        print("Warning: DA calculation received empty array(s).")
        return np.nan

    # Align arrays to the minimum length BEFORE calculating differences
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    y_true_prev = y_true_prev[:min_len]

    # Check again if empty after alignment (shouldn't happen if min_len > 0)
    if len(y_true) == 0: return np.nan

    actual_diff = np.sign(y_true - y_true_prev)
    pred_diff = np.sign(y_pred - y_true_prev) # Compare prediction direction relative to the *same* previous actual

    correct_direction = (actual_diff == pred_diff).astype(int)
    # Avoid division by zero if correct_direction is empty
    if len(correct_direction) == 0: return np.nan

    return np.mean(correct_direction) * 100


# --- Core LSTM Functions ---

def create_multi_output_sequences(X_data, y_data, n_timesteps, n_outputs):
    """
    Creates sequences for multi-output LSTM.
    Assumes X_data and y_data are already appropriately scaled if needed.
    """
    X_seq, y_seq, seq_indices = [], [], []
    # Ensure loop doesn't go out of bounds for features or targets
    max_start_index = len(X_data) - n_timesteps - n_outputs + 1
    if max_start_index <= 0:
        print(f"Warning: Not enough data ({len(X_data)} points) to create sequences with n_timesteps={n_timesteps} and n_outputs={n_outputs}.")
        return np.array(X_seq), np.array(y_seq), np.array(seq_indices)

    for i in range(max_start_index):
        X_seq.append(X_data[i:(i + n_timesteps)])
        y_seq.append(y_data[(i + n_timesteps):(i + n_timesteps + n_outputs)])
        # Store the index of the *end* of the input sequence (last historical point used)
        seq_indices.append(i + n_timesteps - 1)

    return np.array(X_seq), np.array(y_seq), np.array(seq_indices)


def keras_tuner_build_model(hp, n_timesteps, n_features, n_outputs, tune_dropout=True):
    """Build function for Keras Tuner (Matches original)."""
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))

    # Tune LSTM units
    hp_units = hp.Int('units', min_value=8, max_value=128, step=4) # Keep original range
    model.add(LSTM(hp_units, activation='relu', return_sequences=False))

    # Optional Dropout Tuning
    if tune_dropout:
        # Use same name 'dropout' as in original script
        hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(hp_dropout))

    model.add(Dense(n_outputs))

    # Tune learning rate
    # Use same name 'learning_rate' as in original script
    hp_learning_rate = hp.Choice('learning_rate',
                                 values=[1e-3, 5e-3, 5e-4, 5e-5, 1e-4, 1e-5]) # Keep original choices

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse') # Use 'mse' loss as in original
    return model

def run_lstm_walk_forward(
    X_all_df, # Original unscaled features DataFrame with DatetimeIndex
    y_pct_change_all_df, # Original unscaled pct change target Series/DataFrame with DatetimeIndex
    y_level_all_df, # Original unscaled level target Series/DataFrame with DatetimeIndex
    train_end_date_str,
    test_start_date_str,
    final_historical_date_str, # To filter data range
    n_timesteps,
    n_outputs,
    target_level_col, # Name of the level column in y_level_all_df
    tune_dropout=True,
    tuner_max_trials=8,
    tuner_epochs=15,
    final_model_epochs=50,
    lstm_batch_size=16,
    lstm_patience=10,
    tuner_dir='lstm_bo_tuner_wf_dir' # Separate dir for walk-forward tuning steps
    ):
    """
    Runs LSTM walk-forward validation mimicking the original script's logic.

    Handles scaling, sequence creation, walk-forward loop with tuning,
    prediction with inverse scaling, and metric calculation.
    """
    print("--- Starting Utility: run_lstm_walk_forward ---")

    # --- 1. Prepare Data (Filter, Dropna) ---
    final_historical_date = pd.to_datetime(final_historical_date_str)
    # Combine inputs for consistent filtering and NaN dropping
    df_combined = pd.concat([X_all_df, y_pct_change_all_df, y_level_all_df], axis=1)
    df_filtered = df_combined[df_combined.index <= final_historical_date].copy()

    initial_rows = len(df_filtered)
    df_processed = df_filtered.dropna() # Drop rows with any NaN in features or targets
    rows_after_na_drop = len(df_processed)
    print(f"Dropped {initial_rows - rows_after_na_drop} rows containing NaNs.")
    if df_processed.empty: raise ValueError("DataFrame empty after dropping NaNs.")

    feature_cols = X_all_df.columns.tolist()
    target_pct_change_col = y_pct_change_all_df.columns[0] # Assume single target column

    # Extract processed data using the index of df_processed
    X_all = df_processed[feature_cols]
    y_pct_change_all = df_processed[[target_pct_change_col]]
    y_level_all = df_processed[[target_level_col]] # Use the actual level df passed
    n_features = X_all.shape[1]

    # --- 2. Split Data (Temporal) ---
    train_end_date = pd.to_datetime(train_end_date_str)
    test_start_date = pd.to_datetime(test_start_date_str)

    X_train = X_all[X_all.index <= train_end_date]
    y_train_pct_change = y_pct_change_all[y_pct_change_all.index <= train_end_date]
    # y_train_level = y_level_all[y_level_all.index <= train_end_date] # Not directly needed for scaling/seq

    X_test = X_all[X_all.index >= test_start_date]
    y_test_pct_change = y_pct_change_all[y_pct_change_all.index >= test_start_date]
    # y_test_level = y_level_all[y_level_all.index >= test_start_date] # Used later for actuals

    if X_test.empty or X_train.empty: raise ValueError("Training or Test set is empty after split.")
    print(f"Initial training data shape: X={X_train.shape}, y_pct={y_train_pct_change.shape}")
    print(f"Test data shape: X={X_test.shape}, y_pct={y_test_pct_change.shape}")

    # --- 3. Scaling (Fit on Train only, Transform Train & Test) ---
    print("Scaling features and target (fitting only on initial training data)...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test) # Use the same scaler

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_pct_change)
    y_test_scaled = scaler_y.transform(y_test_pct_change) # Use the same scaler
    print("Scaling complete.")

    # --- 4. Create Sequences (Once on the full scaled dataset) ---
    print(f"Creating sequences for the entire dataset (n_timesteps={n_timesteps}, n_outputs={n_outputs})...")
    X_scaled_all = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
    y_scaled_all = np.concatenate((y_train_scaled, y_test_scaled), axis=0)

    # Create sequences using the concatenated scaled data
    X_seq_all, y_seq_all, seq_indices_all = create_multi_output_sequences(
        X_scaled_all, y_scaled_all.flatten(), n_timesteps, n_outputs # Flatten y for sequence creation
    )

    # Get the dates corresponding to the *end* of each input sequence
    # Use the index from the combined X_all dataframe (which is df_processed's index)
    seq_end_dates_all = X_all.index[seq_indices_all]
    print(f"Total sequences created: {len(X_seq_all)}")
    if len(X_seq_all) == 0: raise ValueError("No sequences were created.")

    # --- 5. Split Sequences for Walk-Forward ---
    # Find the index where the test period begins based on the *end date* of the input sequence
    # A sequence belongs to the test set if its input data ends *before* the first prediction date that falls into the test period.
    # Or, simpler: Find the first sequence whose input ends on or after the test_start_date minus the lookback period (or just use test_start_date for simplicity, matching original logic more closely).
    # Original logic seemed based on the *first prediction date* of the sequence.
    # first_pred_dates = X_all.index[seq_indices_all + 1] # Date of the y[0] for each sequence

    # Let's use the sequence end date for splitting, which seems more robust
    # Find the first sequence whose input data ENDS at or after train_end_date
    test_seq_start_index = np.argmax(seq_end_dates_all >= train_end_date)
    # Adjust if the first sequence is already in the test period
    # A better way matching original: Find the first sequence index where the *first predicted output* date falls >= test_start_date
    first_output_indices = seq_indices_all + 1
    # Ensure indices are within bounds of the original DataFrame index
    valid_first_output_indices = first_output_indices[first_output_indices < len(X_all.index)]
    valid_seq_indices = np.where(first_output_indices < len(X_all.index))[0]

    if len(valid_first_output_indices) == 0:
         raise ValueError("Cannot determine test sequence start index: No valid first output indices.")

    first_output_dates = X_all.index[valid_first_output_indices]
    test_seq_mask = first_output_dates >= test_start_date

    if not np.any(test_seq_mask):
         print(f"Warning: No sequences found where the first prediction date is on or after {test_start_date_str}. Using all sequences after initial training end date.")
         # Fallback to using sequences whose input ends after training
         test_seq_mask_fallback = seq_end_dates_all[valid_seq_indices] > train_end_date
         if not np.any(test_seq_mask_fallback):
              raise ValueError("Cannot determine test sequence start index with fallback method.")
         test_seq_start_index = valid_seq_indices[np.argmax(test_seq_mask_fallback)]
    else:
         test_seq_start_index = valid_seq_indices[np.argmax(test_seq_mask)]


    X_seq_train = X_seq_all[:test_seq_start_index]
    y_seq_train = y_seq_all[:test_seq_start_index]
    X_seq_test = X_seq_all[test_seq_start_index:]
    y_seq_test_actual_scaled = y_seq_all[test_seq_start_index:] # Scaled actuals for history update
    test_seq_end_input_dates = seq_end_dates_all[test_seq_start_index:] # Dates for finding last level

    print(f"Sequence Train shapes: X={X_seq_train.shape}, y={y_seq_train.shape}")
    print(f"Sequence Test shapes: X={X_seq_test.shape}, y={y_seq_test_actual_scaled.shape}")
    print(f"Number of test forecast origins: {len(X_seq_test)}")
    if len(X_seq_test) == 0:
        print("Warning: No test sequences generated. Walk-forward validation cannot proceed.")
        return pd.DataFrame(), {} # Return empty results

    # --- 6. Walk-Forward Validation Loop ---
    # History stores SEQUENCES now
    history_X = [x for x in X_seq_train]
    history_y = [y for y in y_seq_train]

    # Store results (levels)
    predictions_level = {h: [] for h in range(1, n_outputs + 1)}
    actuals_level = {h: [] for h in range(1, n_outputs + 1)}
    prev_actuals_level = {h: [] for h in range(1, n_outputs + 1)} # For DA metric
    target_dates_level = {h: [] for h in range(1, n_outputs + 1)} # Dates for plotting/indexing

    print(f"\nStarting walk-forward validation (LSTM with Bayesian Optimization Tuning)...")
    print(f"Using Tuner Directory: {tuner_dir}")
    start_walk_time = time.time()

    # Optional: Clean previous tuner results for this run
    if os.path.exists(tuner_dir):
        print(f"Cleaning previous tuner directory: {tuner_dir}")
        shutil.rmtree(tuner_dir)

    # Infer frequency for date calculations
    freq = pd.infer_freq(X_all.index)
    print(f"Inferred frequency: {freq}")
    date_offset = None
    if freq is None:
        print("Warning: Could not infer frequency. Using fallback DateOffset(months=1). Adjust if needed.")
        # Fallback - adjust if your data isn't monthly
        date_offset = pd.DateOffset(months=1)
    else:
        try:
             date_offset = pd.tseries.frequencies.to_offset(freq)
             print(f"Using offset: {date_offset}")
        except ValueError:
             print(f"Warning: Could not convert inferred frequency '{freq}' to offset. Using fallback DateOffset(months=1).")
             date_offset = pd.DateOffset(months=1)
        # Ensure offset is not None
        if date_offset is None:
            print("Error: Date offset could not be determined. Aborting.")
            raise ValueError("Date offset is None after inference attempt.")


    for t in range(len(X_seq_test)):
        end_input_date = test_seq_end_input_dates[t]
        print(f"\nProcessing Origin {t + 1}/{len(X_seq_test)} (Input Ends: {end_input_date.strftime('%Y-%m-%d')})...")
        start_step_time = time.time()

        current_X_train_seq = np.array(history_X)
        current_y_train_seq = np.array(history_y)

        # --- 6.1 Tune Model ---
        print(f"   Tuning hyperparameters (max_trials={tuner_max_trials})...")
        # Need to pass necessary args to the tuner build function
        build_fn = lambda hp: keras_tuner_build_model(
            hp, n_timesteps, n_features, n_outputs, tune_dropout
            )

        # Unique project name per step to avoid conflicts if not overwriting fully
        project_name = f'lstm_bo_tuning_step_{t+1}'

        tuner = kt.BayesianOptimization(
            build_fn,
            objective='val_loss',
            max_trials=tuner_max_trials,
            executions_per_trial=1,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True # Overwrite specific project for this step
        )

        tuner_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        best_hps = None
        model = None
        try:
            # Check if enough data for validation split
            if len(current_X_train_seq) * 0.2 < 1:
                 val_split = 0.0
                 print("   Warning: Not enough training sequences for validation split during tuning.")
            else:
                 val_split = 0.2

            tuner.search(current_X_train_seq, current_y_train_seq,
                         epochs=tuner_epochs, # Use tuner_epochs
                         batch_size=lstm_batch_size,
                         validation_split=val_split,
                         callbacks=[tuner_early_stopping],
                         verbose=0) # Use verbose=0 like original

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # Safely get hyperparameters for printing
            units_hp = best_hps.get('units')
            lr_hp = best_hps.get('learning_rate')
            dropout_hp = best_hps.get('dropout') if tune_dropout else None
            lr_str = f"{lr_hp:.1e}" if lr_hp is not None else "N/A"
            dropout_str = f", Dropout={dropout_hp:.1f}" if dropout_hp is not None else ""

            print(f"   Best HPs found: Units={units_hp}, LR={lr_str}{dropout_str}")

            # Build the best model found by the tuner
            model = tuner.hypermodel.build(best_hps)

        except Exception as e:
            print(f"   ERROR during Keras Tuner search for step {t+1}: {e}")
            print(f"   Skipping prediction for this origin.")
             # Still update history with the actual data for this step
            history_X.append(X_seq_test[t])
            history_y.append(y_seq_test_actual_scaled[t])
            tf.keras.backend.clear_session() # Clear session memory
            continue # Skip to next iteration

        # --- 6.2 Train Final Model for this step ---
        print(f"   Training final model for step {t+1} (max_epochs={final_model_epochs})...")
        final_early_stopping = EarlyStopping(monitor='loss', patience=lstm_patience, restore_best_weights=True, verbose=0)
        try:
            # Train on the full current history
            history_fit = model.fit(current_X_train_seq, current_y_train_seq,
                      epochs=final_model_epochs, verbose=0, batch_size=lstm_batch_size,
                      callbacks=[final_early_stopping])
            print(f"   Final model training stopped at epoch {len(history_fit.history['loss'])}")

        except Exception as e:
            print(f"   ERROR during final model training for step {t+1}: {e}")
            print(f"   Skipping prediction for this origin.")
            history_X.append(X_seq_test[t])
            history_y.append(y_seq_test_actual_scaled[t])
            tf.keras.backend.clear_session()
            continue

        # --- 6.3 Predict ---
        y_pred_scaled_vector = None
        try:
            # Prepare the single test sequence for prediction
            current_X_test_seq = X_seq_test[t].reshape((1, n_timesteps, n_features))
            # Predict the SCALED percentage changes
            y_pred_scaled_vector = model.predict(current_X_test_seq, verbose=0)[0]

        except Exception as e:
            print(f"   ERROR during prediction for step {t+1}: {e}")
            print(f"   Skipping prediction storage for this origin.")
            history_X.append(X_seq_test[t])
            history_y.append(y_seq_test_actual_scaled[t])
            tf.keras.backend.clear_session()
            continue

        # --- 6.4 Inverse Transform and Store ---
        last_actual_level = None # Initialize for safety
        try:
            # **** CRITICAL: Inverse transform the predictions ****
            y_pred_pct_change_vector = scaler_y.inverse_transform(y_pred_scaled_vector.reshape(1, -1)).flatten()

            # Get the last actual LEVEL known at the end of the input sequence
            # Use the original y_level_all DataFrame (already filtered/processed) indexed by date
            # *** CORRECTED: Remove .iloc[0] ***
            last_actual_level = y_level_all.loc[end_input_date, target_level_col]

        except KeyError:
            print(f"   Warning: Cannot find actual level data for date {end_input_date.strftime('%Y-%m-%d')}. Skipping origin {t+1}.")
            history_X.append(X_seq_test[t])
            history_y.append(y_seq_test_actual_scaled[t])
            tf.keras.backend.clear_session()
            continue
        except Exception as e:
             print(f"   ERROR during inverse transform or level lookup for step {t+1}: {e}")
             print(f"   Skipping prediction storage for this origin.")
             history_X.append(X_seq_test[t])
             history_y.append(y_seq_test_actual_scaled[t])
             tf.keras.backend.clear_session()
             continue

        # Calculate iterative level predictions for h=1 to n_outputs
        current_level_pred = last_actual_level
        current_pred_date = end_input_date

        for h in range(1, n_outputs + 1):
            # Use the inverse-transformed percentage change
            pred_pct_change = y_pred_pct_change_vector[h-1]
            current_level_pred = current_level_pred * (1 + pred_pct_change)

            # Calculate the target date for this forecast horizon
            try:
                target_date = current_pred_date + date_offset # Add offset repeatedly
            except TypeError as te:
                 print(f"   Error adding date offset: current_pred_date={current_pred_date}, date_offset={date_offset}. Error: {te}")
                 # Attempt to handle potential NaT dates if they occur
                 if pd.isna(current_pred_date):
                     print("   Current prediction date is NaT, cannot proceed with horizon calculation.")
                     break # Stop processing horizons for this origin
                 else:
                     # If error is different, re-raise or handle specifically
                     raise te

            # Store prediction
            predictions_level[h].append(current_level_pred)
            target_dates_level[h].append(target_date)

            # Try to find corresponding actual and previous actual levels
            actual_level_h = np.nan # Initialize as NaN
            prev_actual_level_h = np.nan # Initialize as NaN
            valid_actual_found = False
            valid_prev_actual_found = False

            if target_date in y_level_all.index:
                try:
                    # *** CORRECTED: Remove .iloc[0] ***
                    actual_level_h = y_level_all.loc[target_date, target_level_col]
                    valid_actual_found = True
                except Exception as e:
                    print(f"   Warning: Error accessing actual level for target date {target_date}: {e}")
                    valid_actual_found = False # Ensure flag is false

            if valid_actual_found:
                actuals_level[h].append(actual_level_h)
                # Find previous actual date for DA metric
                try:
                     prev_actual_date = target_date - date_offset
                     if prev_actual_date in y_level_all.index:
                         # *** CORRECTED: Remove .iloc[0] ***
                         prev_actual_level_h = y_level_all.loc[prev_actual_date, target_level_col]
                         valid_prev_actual_found = True
                     else:
                         valid_prev_actual_found = False
                         # print(f"   Debug: Prev actual date {prev_actual_date} not in index.")

                except Exception as e:
                     print(f"   Warning: Error calculating or accessing previous actual level date for {target_date}: {e}")
                     valid_prev_actual_found = False

                if valid_prev_actual_found:
                     prev_actuals_level[h].append(prev_actual_level_h)
                else:
                     # If previous actual is missing, we cannot calculate DA accurately. Remove corresponding entries.
                     # print(f"   Debug: Prev actual level not found for date {prev_actual_date} (target {target_date})")
                     actuals_level[h].pop() # Remove the actual added above
                     predictions_level[h].pop() # Remove the prediction
                     target_dates_level[h].pop() # Remove the date
            else:
                 # If actual level for the target date is missing, remove the prediction and date added earlier
                 # print(f"   Debug: Actual level not found for date {target_date}")
                 predictions_level[h].pop()
                 target_dates_level[h].pop()

            # Update the date for the next iteration of h based on the calculated target_date
            current_pred_date = target_date


        # --- 6.5 Update History (append the actual test sequence just used) ---
        history_X.append(X_seq_test[t])
        history_y.append(y_seq_test_actual_scaled[t]) # Append the SCALED actual y

        step_time = time.time() - start_step_time
        print(f"   Step {t + 1} finished in {step_time:.2f} seconds.")
        tf.keras.backend.clear_session() # Clear session memory after each step

    walk_time = time.time() - start_walk_time
    print(f"\nWalk-forward validation complete in {walk_time:.2f} seconds ({walk_time/60:.2f} minutes).")

    # --- 7. Calculate Metrics for Each Horizon ---
    print("\nCalculating performance metrics...")
    final_metrics = []
    plot_dfs = {} # Store dataframes for plotting, similar to original

    for h in range(1, n_outputs + 1):
        preds = np.array(predictions_level[h])
        actuals = np.array(actuals_level[h])
        prev_actuals = np.array(prev_actuals_level[h]) # Previous actuals for DA
        indices = pd.to_datetime(target_dates_level[h])

        # Critical check for alignment before calculating metrics
        min_len = min(len(preds), len(actuals), len(prev_actuals))

        if min_len != len(preds) or min_len != len(actuals) or min_len != len(prev_actuals):
             print(f"Warning: Length mismatch before metrics for h={h}. Preds={len(preds)}, Actuals={len(actuals)}, PrevActuals={len(prev_actuals)}. Using min_len={min_len}")
             # This indicates an issue in the storing logic if lengths don't match here.
             # For safety, trim to minimum length, but investigate the cause.
             if min_len > 0:
                 preds = preds[:min_len]
                 actuals = actuals[:min_len]
                 prev_actuals = prev_actuals[:min_len]
                 indices = indices[:min_len]
             else:
                 print(f"Skipping metrics for h={h} due to zero length after alignment.")
                 preds, actuals, prev_actuals, indices = np.array([]), np.array([]), np.array([]), pd.Index([]) # Ensure empty arrays/index

        if len(preds) > 0: # Check if we have any valid data points for this horizon
            try:
                mse = mean_squared_error(actuals, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actuals, preds)
                # Use the helper functions now part of the utils
                mape = mean_absolute_percentage_error_np(actuals, preds)
                da = directional_accuracy_np(actuals, preds, prev_actuals) # Use helper

                final_metrics.append({'Forecast Horizon (Steps)': h, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'DA': da})
                # Create DataFrame for this horizon's results
                plot_dfs[h] = pd.DataFrame({'Actual_Level': actuals, 'Predicted_Level': preds}, index=indices)
            except Exception as metric_error:
                 print(f"Error calculating metrics for h={h}: {metric_error}")
                 # Append NaN metrics if calculation fails
                 final_metrics.append({'Forecast Horizon (Steps)': h, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'DA': np.nan})
                 plot_dfs[h] = pd.DataFrame() # Add empty DataFrame placeholder

        else:
            # Append NaN metrics if no valid data points
            print(f"No valid predictions/actuals found for horizon h={h} after alignment checks. Skipping metrics.")
            final_metrics.append({'Forecast Horizon (Steps)': h, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'DA': np.nan})
            plot_dfs[h] = pd.DataFrame() # Add empty DataFrame placeholder

    metrics_df = pd.DataFrame(final_metrics)
    # Ensure correct column order as per original script
    metric_cols_order = ['Forecast Horizon (Steps)', 'MSE', 'RMSE', 'MAE', 'MAPE', 'DA']
    existing_cols = [col for col in metric_cols_order if col in metrics_df.columns]
    metrics_df = metrics_df[existing_cols]

    print("--- Finished Utility: run_lstm_walk_forward ---")

    # Return metrics and the dictionary of plot dataframes
    return metrics_df, plot_dfs


# --- Final Forecast Function (Adjusted for Scaling) ---

def final_lstm_forecast(
    X_all_df, # Original unscaled features DataFrame
    y_pct_change_all_df, # Original unscaled pct change target
    y_level_all_df, # Original unscaled level target
    final_train_end_date_str, # Last date of data to use for training
    num_forecast_steps, # How many steps ahead to predict (n_outputs)
    target_level_col, # Name of the level column
    n_timesteps=4,
    tune_dropout=True,
    tuner_max_trials=15,
    tuner_epochs=15,
    final_model_epochs=50,
    lstm_batch_size=16,
    lstm_patience=10,
    tuner_dir='lstm_bo_tuner_final' # Separate dir for final tuning
    ):
    """
    Generates final forecast using LSTM, mimicking the second original script.
    Includes single tuning run, final training, scaling, and inverse scaling.
    """
    print("--- Starting Utility: final_lstm_forecast ---")

    n_outputs = num_forecast_steps # Align parameter name

    # --- 1. Prepare Data (Filter up to final date, Dropna) ---
    final_historical_date = pd.to_datetime(final_train_end_date_str)
    # Combine inputs for consistent filtering and NaN dropping
    df_combined = pd.concat([X_all_df, y_pct_change_all_df, y_level_all_df], axis=1)
    # Ensure filtering uses the correct date column (the index)
    df_filtered = df_combined[df_combined.index <= final_historical_date].copy()
    print(f"Data filtered up to {final_train_end_date_str}. Shape: {df_filtered.shape}")

    initial_rows = len(df_filtered)
    df_processed = df_filtered.dropna()
    rows_after_na_drop = len(df_processed)
    print(f"Dropped {initial_rows - rows_after_na_drop} rows containing NaNs.")
    if df_processed.empty: raise ValueError("DataFrame empty after dropping NaNs.")

    feature_cols = X_all_df.columns.tolist()
    target_pct_change_col = y_pct_change_all_df.columns[0]

    # Define final training data from processed data
    X_train_final = df_processed[feature_cols]
    y_train_pct_change_final = df_processed[[target_pct_change_col]]
    y_train_level_final = df_processed[[target_level_col]] # Use actual levels df from processed data
    n_features = X_train_final.shape[1]
    print(f"Final training data shape: X={X_train_final.shape}, y_pct={y_train_pct_change_final.shape}")

    # --- 2. Final Scaling (Fit and Transform on final training data) ---
    print("\nScaling features and target for final model...")
    final_scaler_X = StandardScaler()
    X_train_final_scaled = final_scaler_X.fit_transform(X_train_final)

    final_scaler_y = StandardScaler()
    y_train_final_scaled = final_scaler_y.fit_transform(y_train_pct_change_final) # Scale y
    print("Scaling complete.")

    # --- 3. Create Sequences for Tuning/Training ---
    print(f"Creating sequences for final training (n_timesteps={n_timesteps}, n_outputs={n_outputs})...")
    if len(X_train_final_scaled) < n_timesteps + n_outputs:
        raise ValueError(f"Not enough data ({len(X_train_final_scaled)} rows) for sequences.")

    X_seq_train_final, y_seq_train_final, _ = create_multi_output_sequences(
        X_train_final_scaled, y_train_final_scaled.flatten(), n_timesteps, n_outputs # Use scaled y
    )
    print(f"Final Training sequences created: X={X_seq_train_final.shape}, y={y_seq_train_final.shape}")
    if X_seq_train_final.shape[0] == 0:
        raise ValueError("No final training sequences generated.")

    # --- 4. Final Hyperparameter Tuning (ONCE) ---
    print(f"\nTuning hyperparameters ONCE (max_trials={tuner_max_trials})...")
    # Optional: Clean previous results
    if os.path.exists(tuner_dir):
        print(f"Cleaning previous tuner directory: {tuner_dir}")
        shutil.rmtree(tuner_dir)

    build_fn = lambda hp: keras_tuner_build_model(
        hp, n_timesteps, n_features, n_outputs, tune_dropout
        )
    tuner = kt.BayesianOptimization(
        build_fn,
        objective='val_loss',
        max_trials=tuner_max_trials,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name='lstm_final_tuning',
        overwrite=True
    )

    tuner_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0) # Use patience=5 like original
    tuning_start_time = time.time()
    best_hps = None
    final_model = None

    try:
        # Check for validation split possibility
        if len(X_seq_train_final) * 0.2 < 1:
             val_split = 0.0
             print("   Warning: Not enough final sequences for validation split during tuning.")
        else:
             val_split = 0.2

        tuner.search(X_seq_train_final, y_seq_train_final,
                     epochs=tuner_epochs, # Use tuner_epochs
                     batch_size=lstm_batch_size,
                     validation_split=val_split,
                     callbacks=[tuner_early_stopping],
                     verbose=1) # Show tuner progress like original

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\nFinal Tuning Complete ({time.time() - tuning_start_time:.2f}s).")
        # Safely print HPs
        units_hp_final = best_hps.get('units')
        lr_hp_final = best_hps.get('learning_rate')
        dropout_hp_final = best_hps.get('dropout') if tune_dropout else None
        lr_str_final = f"{lr_hp_final:.1e}" if lr_hp_final is not None else "N/A"
        dropout_str_final = f", Dropout={dropout_hp_final:.1f}" if dropout_hp_final is not None else ""
        print(f"   Best HPs found: Units={units_hp_final}, LR={lr_str_final}{dropout_str_final}")


        final_model = tuner.hypermodel.build(best_hps)

    except Exception as e:
        print(f"   ERROR during final Keras Tuner search: {e}")
        tf.keras.backend.clear_session()
        raise # Stop execution if final tuning fails

    # --- 5. Train Final Model ---
    print(f"\nTraining final model with best HPs (max_epochs={final_model_epochs})...")
    final_early_stopping = EarlyStopping(monitor='loss', patience=lstm_patience, restore_best_weights=True, verbose=0)
    training_start_time = time.time()
    try:
        history_fit_final = final_model.fit(X_seq_train_final, y_seq_train_final, # Train on all final sequences
                  epochs=final_model_epochs, verbose=1, batch_size=lstm_batch_size, # verbose=1 like original
                  callbacks=[final_early_stopping])
        print(f"Final model training finished ({time.time() - training_start_time:.2f}s). Stopped at epoch {len(history_fit_final.history['loss'])}")
    except Exception as e:
        print(f"   ERROR during final model training: {e}")
        tf.keras.backend.clear_session()
        raise

    # --- 6. Prepare Input for Forecasting ---
    print("\nPreparing input sequence for forecast...")
    last_sequence_input_scaled = X_train_final_scaled[-n_timesteps:]
    forecast_input = last_sequence_input_scaled.reshape((1, n_timesteps, n_features))
    print(f"Forecast input shape: {forecast_input.shape}")

    # Use y_train_level_final which comes from the processed data
    last_known_level = y_train_level_final.iloc[-1].loc[target_level_col] # Get scalar value correctly
    last_date = y_train_level_final.index[-1]
    print(f"Last known actual level ({last_date.strftime('%Y-%m-%d')}): {last_known_level:.4f}")

    # --- 7. Generate Forecast ---
    print(f"Generating {num_forecast_steps}-step forecast...")
    y_pred_scaled_vector = None
    y_pred_pct_change_vector = None
    try:
        # Predict the vector of SCALED percentage changes
        y_pred_scaled_vector = final_model.predict(forecast_input, verbose=0)[0]

        # **** CRITICAL: Inverse scale the predictions ****
        y_pred_pct_change_vector = final_scaler_y.inverse_transform(y_pred_scaled_vector.reshape(1, -1)).flatten()

    except Exception as e:
        print(f"   ERROR during forecast prediction or inverse transform: {e}")
        tf.keras.backend.clear_session()
        raise

    # --- 8. Calculate Forecast Levels and Dates ---
    forecast_results = []
    current_pred_level = last_known_level

    # Determine date frequency from the final training data index
    inferred_freq = pd.infer_freq(X_train_final.index)
    print(f"Inferred data frequency: {inferred_freq}")
    date_offset = None
    if inferred_freq is None:
        print("Warning: Could not infer frequency. Using fallback DateOffset(months=1). Adjust if needed.")
        date_offset = pd.DateOffset(months=1)
    else:
        try:
            date_offset = pd.tseries.frequencies.to_offset(inferred_freq)
            print(f"Using offset: {date_offset}")
        except ValueError:
             print(f"Warning: Could not convert inferred frequency '{inferred_freq}' to offset. Using fallback DateOffset(months=1).")
             date_offset = pd.DateOffset(months=1)
        # Ensure offset is not None
        if date_offset is None:
            print("Error: Date offset could not be determined. Aborting.")
            raise ValueError("Date offset is None after inference attempt.")


    current_pred_date = last_date

    for step in range(1, num_forecast_steps + 1):
        # Use the inverse-transformed percentage change
        pred_pct_change = y_pred_pct_change_vector[step-1]
        current_pred_level = current_pred_level * (1 + pred_pct_change)

        # Calculate forecast date
        try:
            current_pred_date = current_pred_date + date_offset
        except TypeError as te:
             print(f"   Error adding date offset: current_pred_date={current_pred_date}, date_offset={date_offset}. Error: {te}")
             break # Stop forecast generation if date calculation fails

        forecast_results.append({
            'Date': current_pred_date, # Store as datetime object initially
            'LSTM_Forecast': current_pred_level
        })

    # --- 9. Format Forecast Output ---
    forecast_df = pd.DataFrame(forecast_results)
    # Set index only if results were generated
    if not forecast_df.empty:
        forecast_df = forecast_df.set_index('Date')

    print("--- Finished Utility: final_lstm_forecast ---")

    # Return forecast, the trained model, and best HPs found
    # Returning the scalers might also be useful for external analysis
    return forecast_df, final_model, best_hps, final_scaler_X, final_scaler_y