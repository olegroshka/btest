import os
import tensorflow as tf
from tkan import TKAN  # Ensure TKAN is correctly installed and compatible with your TensorFlow version
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def download_and_update_lvc_data(data_file_path, ticker="LVC.PA", update_days=30):
    """
    Downloads latest LVC data from yfinance and updates the existing Excel file.
    Correctly maps Yahoo Finance columns to existing data structure.
    
    Args:
        data_file_path (str): Path to the existing Excel file
        ticker (str): Yahoo Finance ticker symbol
        update_days (int): Number of days to look back for updates
        
    Returns:
        str: Path to the updated file
    """
    print(f"🔄 Updating LVC data from {ticker}...")
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Load existing data to understand structure
        if not os.path.exists(data_file_path):
            print("❌ No existing data found - cannot update without original!")
            return data_file_path, False
            
        print(f"📂 Loading existing data from: {data_file_path}")
        existing_data = pd.read_excel(data_file_path)
        
        # Store original structure
        original_columns = existing_data.columns.tolist()
        print(f"📋 Original columns: {original_columns}")
        
        # Convert dates and determine structure
        existing_data['Date'] = pd.to_datetime(existing_data['Date'])
        last_date = existing_data['Date'].max()
        first_date = existing_data['Date'].min()
        
        # Check original sort order
        is_descending = existing_data['Date'].iloc[0] > existing_data['Date'].iloc[-1]
        print(f"📅 Original data: {first_date.date()} to {last_date.date()}")
        print(f"📊 Original records: {len(existing_data)}")
        print(f"🔽 Sort order: {'DESCENDING' if is_descending else 'ASCENDING'}")
        
        # METHOD 1: Try web scraping first (PRIMARY METHOD)
        print("🌐 Attempting web scraping (primary method)...")
        try:
            from scrape_yahoo_finance import scrape_yahoo_finance_history
            
            scraped_data = scrape_yahoo_finance_history(ticker)
            
            if not scraped_data.empty:
                print(f"✅ Web scraping successful - downloaded {len(scraped_data)} records")
                
                # Find only truly new dates
                existing_dates = set(existing_data['Date'].dt.date)
                new_dates_only = scraped_data[~scraped_data['Date'].dt.date.isin(existing_dates)]
                
                if new_dates_only.empty:
                    print("✅ No new dates to add - data is up to date")
                    return data_file_path, False
                
                print(f"🆕 Found {len(new_dates_only)} new dates to add")
                
                # Map scraped data to existing structure
                new_records = []
                for _, row in new_dates_only.iterrows():
                    new_record = {}
                    for col in original_columns:
                        if col == 'Date':
                            new_record[col] = row['Date']
                        elif col == 'Close':
                            new_record[col] = row['Close']
                        elif col == 'High':
                            new_record[col] = row['High'] 
                        elif col == 'Low':
                            new_record[col] = row['Low']
                        elif col == 'Volume':
                            new_record[col] = row['Volume']
                        elif col == 'SMAVG (15)':
                            new_record[col] = None
                        else:
                            new_record[col] = None
                    new_records.append(new_record)
                
                # Combine with existing data
                if new_records:
                    new_df = pd.DataFrame(new_records)
                    print(f"📋 New data columns: {new_df.columns.tolist()}")
                    combined_data = pd.concat([existing_data, new_df], ignore_index=True)
                    print(f"📊 Combined data shape: {combined_data.shape}")
                else:
                    combined_data = existing_data.copy()
                
                # Sort data properly (ascending for calculations)
                combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
                
                # Recalculate SMAVG (15) for all data
                if 'SMAVG (15)' in combined_data.columns:
                    print("🔢 Recalculating SMAVG (15)...")
                    combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
                
                # Restore original sort order
                if is_descending:
                    combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
                    print("🔽 Restored DESCENDING sort order")
                else:
                    combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
                    print("🔼 Keeping ASCENDING sort order")
                
                # Ensure column order matches original
                combined_data = combined_data[original_columns]
                
                # Save the updated file
                combined_data.to_excel(data_file_path, index=False)
                
                print(f"✅ Updated data saved to: {data_file_path}")
                print(f"📊 Final records: {len(combined_data)}")
                print(f"📅 Date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
                print(f"🆕 New records added: {len(new_records)}")
                print(f"🔧 Method used: web_scraping")
                
                return data_file_path, len(new_records) > 0
            else:
                print("⚠️ Web scraping returned no data, trying Yahoo Finance API...")
                raise Exception("Web scraping failed")
                
        except Exception as web_error:
            print(f"⚠️ Web scraping failed: {web_error}")
            print("📡 Falling back to Yahoo Finance API...")
        
        # METHOD 2: Fallback to Yahoo Finance API
        start_date = (last_date - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"⬇️ Downloading via API from {start_date} to {end_date}")
        
        yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if yf_data.empty:
            print("⚠️  No new data available from Yahoo Finance")
            return data_file_path, False
            
        print(f"📥 Downloaded {len(yf_data)} records from Yahoo Finance")
        
        # Reset index to get Date as column
        yf_data.reset_index(inplace=True)
        
        # Flatten multi-index columns if they exist
        if yf_data.columns.nlevels > 1:
            yf_data.columns = yf_data.columns.droplevel(1)
        
        yf_data['Date'] = pd.to_datetime(yf_data['Date'])
        
        # Find only truly new dates
        existing_dates = set(existing_data['Date'].dt.date)
        new_dates_only = yf_data[~yf_data['Date'].dt.date.isin(existing_dates)]
        
        if new_dates_only.empty:
            print("✅ No new dates to add - data is up to date")
            return data_file_path, False
        
        print(f"🆕 Found {len(new_dates_only)} new dates to add")
        
        # Map Yahoo Finance columns to existing structure
        new_records = []
        for _, yf_row in new_dates_only.iterrows():
            new_record = {}
            
            # Map each column from Yahoo Finance to existing structure
            for col in original_columns:
                if col == 'Date':
                    new_record[col] = yf_row['Date']
                elif col == 'Close':
                    new_record[col] = yf_row['Close']
                elif col == 'High':
                    new_record[col] = yf_row['High'] 
                elif col == 'Low':
                    new_record[col] = yf_row['Low']
                elif col == 'Volume':
                    new_record[col] = yf_row['Volume']
                elif col == 'SMAVG (15)':
                    # Will be calculated later
                    new_record[col] = None
                else:
                    # Handle any other columns
                    new_record[col] = None
                    
            new_records.append(new_record)
        
        # Combine with existing data
        if new_records:
            new_df = pd.DataFrame(new_records)
            print(f"📋 New data columns: {new_df.columns.tolist()}")
            combined_data = pd.concat([existing_data, new_df], ignore_index=True)
            print(f"📊 Combined data shape: {combined_data.shape}")
        else:
            combined_data = existing_data.copy()
        
        # Sort data properly (ascending for calculations)
        combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Recalculate SMAVG (15) for all data
        if 'SMAVG (15)' in combined_data.columns:
            print("🔢 Recalculating SMAVG (15)...")
            combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
        
        # Restore original sort order
        if is_descending:
            combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
            print("🔽 Restored DESCENDING sort order")
        else:
            combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
            print("🔼 Keeping ASCENDING sort order")
        
        # Ensure column order matches original
        combined_data = combined_data[original_columns]
        
        # Save the updated file
        combined_data.to_excel(data_file_path, index=False)
        
        print(f"✅ Updated data saved to: {data_file_path}")
        print(f"📊 Final records: {len(combined_data)}")
        print(f"📅 Date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
        print(f"🆕 New records added: {len(new_records)}")
        
        return data_file_path, len(new_records) > 0
        
    except ImportError as e:
        print(f"⚠️  Could not import yfinance: {e}")
        print(f"⚠️  Continuing with existing data file...")
        return data_file_path, False
    except Exception as e:
        print(f"⚠️  Error updating data with Yahoo Finance API: {e}")
        print("🌐 Trying web scraping as fallback...")
        
        # Try web scraping fallback
        try:
            from scrape_yahoo_finance import update_lvc_with_scraped_data
            print("📡 Using web scraper to get latest data...")
            result = update_lvc_with_scraped_data(data_file_path, ticker)
            # Web scraper returns just path, assume data may have changed
            return result, True
        except Exception as scrape_error:
            print(f"⚠️  Web scraping also failed: {scrape_error}")
            print(f"⚠️  Continuing with existing data file...")
            return data_file_path, False


def process_stock(data_lvc_path, auto_update=True):
    """
    Processes the stock data and returns the processed DataFrame along with features and target.

    Args:
        data_lvc_path (str): Path to the Excel file containing stock data.
        auto_update (bool): Whether to automatically update data from yfinance

    Returns:
        X (DataFrame): Feature DataFrame.
        y (Series): Target variable Series.
        market_data (DataFrame): Processed market data DataFrame.
        has_new_data (bool): Whether new data was added during update.
    """
    # Auto-update data if requested
    has_new_data = False
    if auto_update:
        data_lvc_path, has_new_data = download_and_update_lvc_data(data_lvc_path)
    
    # Load the data file
    market_data = pd.read_excel(data_lvc_path)

    # Convert 'Date' to datetime and sort in ascending order
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.sort_values('Date')  # Ensure data is sorted oldest first
    market_data.set_index('Date', inplace=True)

    # Ensure SMAVG (15) exists, calculate if missing
    if 'SMAVG (15)' not in market_data.columns:
        print("Calculating SMAVG (15) as it's missing from data...")
        market_data['SMAVG (15)'] = market_data['Close'].rolling(window=15, min_periods=1).mean()

    # Create lagged features
    market_data['Prior Close Price'] = market_data['Close'].shift(1)
    market_data['Prior Volume'] = market_data['Volume'].shift(1)
    market_data['Prior High'] = market_data['High'].shift(1)
    market_data['Prior Low'] = market_data['Low'].shift(1)
    market_data['Prior SMAVG (15)'] = market_data['SMAVG (15)'].shift(1)

    # Drop rows with NaN values created by the shift
    market_data = market_data.dropna()

    # Verify that the DataFrame is not empty
    if market_data.empty:
        raise ValueError(
            "After creating lagged features and dropping NaNs, the DataFrame is empty. Check your data and window size.")

    # Separate features and target
    X = market_data[['Prior Close Price', 'Prior Volume', 'Prior High', 'Prior Low', 'Prior SMAVG (15)']]
    y = market_data['Close']

    return X, y, market_data, has_new_data



def create_sequences(X, y, window_size):
    """
    Creates sequences of data for model training and testing.

    Args:
        X (ndarray): Scaled feature array.
        y (ndarray): Scaled target array.
        window_size (int): Number of past days to consider for each sequence.

    Returns:
        X_seq (ndarray): Feature sequences.
        y_seq (ndarray): Target sequences.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i:i + window_size])  # Predict 10 steps ahead
    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),  # (window_size, 5)

        # TKAN Layers
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 1
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 2
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 3

        tf.keras.layers.Dense(1)  # Output layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the TKAN model.

    Args:
        model (tf.keras.Model): Compiled TKAN model.
        X_train (ndarray): Training feature sequences.
        y_train (ndarray): Training target sequences.
        X_val (ndarray): Validation feature sequences.
        y_val (ndarray): Validation target sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.

    Returns:
        history (History): Training history object.
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history

def create_results_df(scaler_y, y_train_seq, y_test_seq, train_predictions, test_predictions, train_dates, test_dates):
    """
    Creates DataFrames for training and testing results with actual and predicted values.

    Args:
        scaler_y: Scaler fitted on target variable.
        y_train_seq: Scaled training target sequences.
        y_test_seq: Scaled testing target sequences.
        train_predictions: Model predictions on training data.
        test_predictions: Model predictions on testing data.
        train_dates: Dates corresponding to training data.
        test_dates: Dates corresponding to testing data.

    Returns:
        combined_results_df: Combined DataFrame containing both training and testing results.
    """
    # Inverse transform the scaled predictions and actual values
    y_test_actual_scaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_train_actual_scaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).reshape(y_train_seq.shape)
    test_predictions_scaled = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).reshape(
        test_predictions.shape)
    train_predictions_scaled = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).reshape(
        train_predictions.shape)

    # Create a DataFrame for test data to store actual, predicted, and corresponding dates
    results_df = pd.DataFrame({
        'Date': test_dates[:len(test_predictions_scaled)]
    })

    # Add actual close prices for the first day (test set)
    results_df['Actual Close Price'] = y_test_actual_scaled[:, 0]

    # Add predictions for the current day and the next 9 days (test set)
    for i in range(10):
        results_df[f'Predicted Day {i + 1}'] = test_predictions_scaled[:, i]

    # Create a DataFrame for training data to store actual, predicted, and corresponding dates
    train_results_df = pd.DataFrame({
        'Date': train_dates[:len(train_predictions_scaled)]
    })

    # Add actual close prices for the first day (train set)
    train_results_df['Actual Close Price'] = y_train_actual_scaled[:, 0]

    # Add predictions for the current day and the next 9 days (train set)
    for i in range(10):
        train_results_df[f'Predicted Day {i + 1}'] = train_predictions_scaled[:, i]

    # Combine test and train results
    combined_results_df = pd.concat([train_results_df, results_df], ignore_index=True)

    return combined_results_df


def plot_predictions(results_df, start_date, end_date, day):
    """
    Plots actual vs predicted values for a given date range and prediction day.

    Args:
        results_df: DataFrame containing actual and predicted values.
        start_date: Start date for the plot (string in 'YYYY-MM-DD' format).
        end_date: End date for the plot (string in 'YYYY-MM-DD' format).
        day: Prediction day number (1 to 10).
    """
    # Validate the day parameter
    if day < 1 or day > 10:
        raise ValueError("day must be between 1 and 10, representing prediction for Day 1 to Day 10.")

    # Convert start and end dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame for the given date range
    filtered_df = results_df[
        (results_df['Date'] >= start_date) &
        (results_df['Date'] <= end_date)
        ]

    if filtered_df.empty:
        print(f"No data available between {start_date.date()} and {end_date.date()}.")
        return

    # Plot actual vs predicted values for the selected prediction day
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Date'], filtered_df['Actual Close Price'], label='Actual Closing Price', color='blue')
    plt.plot(filtered_df['Date'], filtered_df[f'Predicted Day {day}'], label=f'Predicted Close Price (Day {day})',
             color='red')
    plt.title(f'Actual vs Predicted Closing Prices (Day {day}) from {start_date.date()} to {end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def predict_for_most_recent_date(model, X, scaler_X, scaler_y, market_data, window_size=10, prediction_days=10):
    """
    Predicts the next 'prediction_days' of closing prices based on the most recent observations.

    Args:
        model: Trained TensorFlow model.
        X: Original feature DataFrame.
        scaler_X: Scaler fitted on training features.
        scaler_y: Scaler fitted on training target.
        market_data: Processed market data DataFrame.
        window_size: Number of past days to consider for prediction.
        prediction_days: Number of days to predict ahead.

    Returns:
        future_df: DataFrame containing future dates and predicted close prices.
    """
    # Extract the most recent 'window_size' observations
    X_last_window = X.iloc[-window_size:]

    # Ensure data is sorted in ascending order
    X_last_window = X_last_window.sort_index()

    # Scale the data
    X_last_window_scaled = scaler_X.transform(X_last_window)

    # Reshape to match the input shape of the model
    X_last_window_scaled = X_last_window_scaled.reshape(1, window_size, -1)  # Shape: (1, window_size, 5)

    # Predict the next 'prediction_days' days
    future_predictions_scaled = model.predict(X_last_window_scaled).flatten()

    # Inverse transform the predictions
    future_predictions_actual = scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()

    # Generate future dates
    last_date = market_data.index[-1]
    future_dates = [last_date + BDay(i + 1) for i in range(prediction_days)]  # Next 'prediction_days' business days

    # Create DataFrame
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': future_predictions_actual
    })

    return future_df


def plot_past_and_future(results_df, market_data, future_predictions_df, window_size=20, signal_df=None):
    """
    Plots the past actual close prices and future predicted close prices.

    Args:
        results_df (DataFrame): DataFrame containing both training and testing actual and predicted close prices.
        market_data (DataFrame): Original market data DataFrame.
        future_predictions_df (DataFrame): DataFrame containing future dates and predicted close prices.
        window_size (int): Number of past days to display.
        signal_df (DataFrame, optional): DataFrame containing investment signals.

    Returns:
        combined_df (DataFrame): Combined DataFrame of past actual and future predicted close prices.
    """
    # Extract the most recent 'window_size' days of actual close prices
    recent_actual_df = market_data[['Close']].iloc[-window_size:]

    # Plot past actual close prices and future predicted close prices
    plt.figure(figsize=(12, 6))
    plt.plot(recent_actual_df.index, recent_actual_df['Close'], label='Past Actual Close Price', color='blue')
    plt.plot(future_predictions_df['Date'], future_predictions_df['Predicted Close Price'],
             label='Future Predicted Close Price', color='red')

    # If there are investment signals, annotate them
    if signal_df is not None and not signal_df.empty:
        for _, row in signal_df.iterrows():
            plt.scatter(row['Date'], row['Predicted Close Price'], marker='*', color='gold', s=200, label='Investment Signal')
            plt.annotate('Signal', xy=(row['Date'], row['Predicted Close Price']),
                         xytext=(row['Date'], row['Predicted Close Price'] + 2),
                         arrowprops=dict(facecolor='green', shrink=0.05))

    plt.title(f'Past {window_size} Days and Next {len(future_predictions_df)} Days Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (£)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Combine past and future data
    combined_df = pd.concat([recent_actual_df, future_predictions_df.set_index('Date')])
    combined_df = combined_df.reset_index().rename(columns={'index': 'Date'})

    return combined_df


def evaluate_prediction_on_date(model, scaler_X, scaler_y, X, market_data, selected_date, window_size=10,
                                prediction_days=10):
    """
    Makes predictions for a selected date and compares with actual close prices.

    Args:
        model: Trained TensorFlow model.
        scaler_X: Scaler fitted on training features.
        scaler_y: Scaler fitted on training target.
        X: Original feature DataFrame.
        market_data: Processed market data DataFrame.
        selected_date: The date for which to make predictions (string in 'YYYY-MM-DD' format).
        window_size: Number of past days to consider for prediction.
        prediction_days: Number of days to predict ahead.

    Returns:
        comparison_df: DataFrame containing predicted and actual close prices.
        mae: Mean Absolute Error of the predictions.
        rmse: Root Mean Squared Error of the predictions.
    """
    # Convert selected_date to datetime
    selected_date = pd.to_datetime(selected_date)

    # Check if selected_date is within the market_data index
    if selected_date not in market_data.index:
        raise ValueError(f"The selected date {selected_date.date()} is not in the dataset.")

    # Ensure there are enough past observations
    idx = market_data.index.get_loc(selected_date)
    if idx < window_size:
        raise ValueError(f"Not enough data before {selected_date.date()} to create a window of size {window_size}.")

    # Extract the window_size days before the selected_date
    X_window = X.iloc[idx - window_size:idx]

    # Scale the data
    X_window_scaled = scaler_X.transform(X_window)

    # Reshape to match model input
    X_window_scaled = X_window_scaled.reshape(1, window_size, -1)  # Shape: (1, window_size, 5)

    # Make predictions
    predictions_scaled = model.predict(X_window_scaled).flatten()

    # Inverse transform predictions
    predictions_actual = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Generate predicted dates
    future_dates = [selected_date + BDay(i + 1) for i in range(prediction_days)]

    # Retrieve actual close prices for the predicted dates
    actual_prices = []
    for date in future_dates:
        if date in market_data.index:
            actual_prices.append(market_data.loc[date, 'Close'])
        else:
            actual_prices.append(np.nan)  # Handle missing dates (e.g., weekends, holidays)

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predictions_actual,
        'Actual Close Price': actual_prices
    })

    # Drop rows where actual price is NaN
    comparison_df.dropna(inplace=True)

    # Calculate error metrics
    mae = mean_absolute_error(comparison_df['Actual Close Price'], comparison_df['Predicted Close Price'])
    rmse = np.sqrt(mean_squared_error(comparison_df['Actual Close Price'], comparison_df['Predicted Close Price']))

    print(f"\nPrediction for date: {selected_date.date()}")
    print(comparison_df)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return comparison_df, mae, rmse


def save_investment_signal_to_csv(signal_df, last_actual_close_price, threshold=1.0):
    """
    Saves investment signals to both ForgeFolio and local signal CSV files.
    Only saves ONE signal per day for LVC to avoid duplicates.
    
    Args:
        signal_df (DataFrame): DataFrame containing investment signals with Date and Predicted Close Price
        last_actual_close_price (float): Current actual close price
        threshold (float): Minimum price increase threshold that triggered the signal
    """
    if signal_df.empty:
        print("🚫 No signals to save to CSV files")
        return
    
    import csv
    from datetime import datetime
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forgefolio_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "ForgeFolio", "data", "Andrey", "strategies", "S08", "signal.csv")
    local_path = os.path.join(script_dir, "Data", "signal.csv")
    
    # Prepare signal data
    ticker = "LVC"
    signal_date = datetime.now().strftime('%Y-%m-%d')
    signal_value = 1  # Positive signal (buy)
    
    # Check for duplicate signals today
    def check_duplicate_signal(file_path, ticker, signal_date):
        """Check if signal already exists for this ticker and date"""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] == ticker and row[1] == signal_date:
                        return True
        except Exception:
            pass
        return False
    
    # Check if signal already exists for today
    if check_duplicate_signal(local_path, ticker, signal_date):
        print(f"⚠️ Signal for {ticker} on {signal_date} already exists - skipping duplicate")
        return
    
    # Calculate confidence based on prediction strength
    max_predicted_price = signal_df['Predicted Close Price'].max()
    price_increase = max_predicted_price - last_actual_close_price
    confidence = min(0.95, 0.5 + (price_increase / threshold) * 0.3)  # Scale confidence 0.5-0.95
    confidence = round(confidence, 2)
    
    source = "TKAN_model"
    
    signal_row = [ticker, signal_date, signal_value, confidence, source]
    
    # Save to both files
    for file_path, file_name in [(forgefolio_path, "ForgeFolio"), (local_path, "Local")]:
        try:
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(file_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to CSV
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writerow(['ticker', 'date', 'signal', 'confidence', 'source'])
                
                writer.writerow(signal_row)
            
            print(f"✅ Signal saved to {file_name} file: {file_path}")
            print(f"   📊 Signal: {ticker} | Date: {signal_date} | Signal: {signal_value} | Confidence: {confidence} | Source: {source}")
            
        except Exception as e:
            print(f"❌ Error saving signal to {file_name} file ({file_path}): {e}")
    
    # Also save detailed signal information
    detailed_path = local_path.replace('.csv', '_detailed.csv')
    try:
        # Prepare detailed signal data
        detailed_data = signal_df.copy()
        detailed_data['Ticker'] = ticker
        detailed_data['Signal_Date'] = signal_date
        detailed_data['Signal_Value'] = signal_value
        detailed_data['Confidence'] = confidence
        detailed_data['Source'] = source
        detailed_data['Last_Actual_Price'] = last_actual_close_price
        detailed_data['Price_Increase'] = detailed_data['Predicted Close Price'] - last_actual_close_price
        detailed_data['Threshold_Used'] = threshold
        
        # Reorder columns
        detail_cols = ['Ticker', 'Signal_Date', 'Date', 'Signal_Value', 'Confidence', 'Source', 
                      'Last_Actual_Price', 'Predicted Close Price', 'Price_Increase', 'Threshold_Used']
        detailed_data = detailed_data[detail_cols]
        
        # Save detailed data
        file_exists = os.path.exists(detailed_path)
        detailed_data.to_csv(detailed_path, mode='a', header=not file_exists, index=False)
        
        print(f"✅ Detailed signal data saved to: {detailed_path}")
        
    except Exception as e:
        print(f"❌ Error saving detailed signal data: {e}")


def flag_investment_signal(future_predictions_df, last_actual_close_price, threshold=1.0):
    """
    Checks if any of the predicted days have a closing price increase of at least 'threshold' from the last actual close price.
    Now also saves signals to CSV files when detected.

    Args:
        future_predictions_df (DataFrame): DataFrame containing future dates and predicted close prices.
        last_actual_close_price (float): The actual closing price on the last known date.
        threshold (float): The minimum price increase to consider as a signal.

    Returns:
        signal_df (DataFrame): DataFrame containing the days that meet the investment criteria.
        has_signal (bool): Whether any signal was found.
    """
    # Define the target price
    target_price = last_actual_close_price + threshold

    # Flag the days where predicted close price meets or exceeds the target
    signal_df = future_predictions_df[future_predictions_df['Predicted Close Price'] >= target_price]

    has_signal = not signal_df.empty

    if has_signal:
        print(
            f"\n🚨 Investment Signal Detected! Predicted price meets/exceeds the target of £{target_price:.2f} on the following day(s):")
        print(signal_df[['Date', 'Predicted Close Price']])
        
        # Save only ONE signal for the EARLIEST qualifying day (not all days)
        earliest_signal = signal_df.iloc[0:1]  # Take only the first (earliest) signal
        save_investment_signal_to_csv(earliest_signal, last_actual_close_price, threshold)
        
    else:
        print(
            f"\n📊 No Investment Signal Detected. No predicted close price meets/exceeds the target of £{target_price:.2f} in the next {len(future_predictions_df)} days.")

    return signal_df, has_signal

def review_actual_vs_saved_signals(signal_df, market_data):
    """
    Reviews the actual closing prices against the saved investment signals.

    Args:
        signal_df (DataFrame): DataFrame containing the investment signals.
        market_data (DataFrame): Processed market data DataFrame.
    """
    if signal_df.empty:
        print("No investment signals to review.")
        return

    # Iterate over each signal and compare with actual prices
    print("\nReviewing Actual vs Saved Investment Signals:")
    for _, row in signal_df.iterrows():
        date = row['Date']
        predicted_price = row['Predicted Close Price']
        actual_price = market_data.loc[date, 'Close'] if date in market_data.index else np.nan

        if not np.isnan(actual_price):
            result = "Hit" if actual_price >= predicted_price else "Miss"
            print(f"Date: {date.date()} | Predicted: £{predicted_price:.2f} | Actual: £{actual_price:.2f} | {result}")
        else:
            print(f"Date: {date.date()} | Predicted: £{predicted_price:.2f} | Actual: N/A | Data Missing")


def save_model_weights(model, weights_path):
    """
    Saves the model's weights to the specified path.

    Args:
        model (tf.keras.Model): Trained TensorFlow model.
        weights_path (str): Path to save the weights file.
    """
    model.save_weights(weights_path)
    print(f"Model weights saved to '{weights_path}'.")


def load_model_weights(model, weights_path):
    """
    Loads the model's weights from the specified path.

    Args:
        model (tf.keras.Model): TensorFlow model to load weights into.
        weights_path (str): Path to the weights file.
        
    Returns:
        bool: True if weights loaded successfully, False otherwise.
    """
    try:
        model.load_weights(weights_path)
        print(f"✅ Model weights loaded from '{weights_path}'.")
        return True
    except (ValueError, Exception) as e:
        print(f"⚠️ Failed to load weights: {e}")
        print(f"💡 The weights file appears to be from a different model architecture.")
        print(f"🔄 Model will need to be retrained.")
        return False


def main_script():
    """
    Main function to execute the stock prediction and signal flagging.
    """
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_lvc_path = os.path.join(script_dir, "..", "..", "Data", "Archive", "LVC_daily.xlsx")
    weights_path = os.path.join(script_dir, "tkan_model_weights.weights.h5")
    retrain = False  # Set to True to retrain the model and overwrite weights; False to load existing weights if available
    auto_update_data = True  # Enabled with web scraping (primary) + Yahoo Finance API fallback
    window_size = 10
    prediction_days = 10
    threshold = 1.0  # £1 increase threshold for investment signal
    selected_date = '2025-03-31'  # Replace with your desired date in 'YYYY-MM-DD' format

    print("=== TKAN Stock Prediction Pipeline ===")
    print(f"Auto-update data: {auto_update_data}")
    print(f"Retrain model: {retrain}")
    print(f"Data file: {data_lvc_path}")
    
    # Process the stock data (with optional auto-update)
    X, y, market_data, has_new_data = process_stock(data_lvc_path, auto_update=auto_update_data)
    
    # Auto-retrain if new data was added
    if has_new_data and not retrain:
        print("\n🔄 NEW DATA DETECTED - Model will be retrained automatically")
        retrain = True

    # Normalize features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    # Verify that sequences are created
    if len(X_seq) == 0:
        raise ValueError(f"No sequences created with window_size={window_size}. Check if your dataset is large enough.")

    # Split into training and testing sets
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, train_dates, test_dates = train_test_split(
        X_seq, y_seq, market_data.index[window_size:], test_size=0.05, shuffle=False
    )

    # Build the model
    input_shape = X_train_seq.shape[1:]  # (window_size, number_of_features)
    model = build_model(input_shape)

    # Force the model to build by specifying the input shape
    model.build((None, *input_shape))  # (batch_size, window_size, number_of_features))
    print("Model built successfully.")

    # Check if weights exist and load or train accordingly
    weights_loaded = False
    if not retrain and os.path.exists(weights_path):
        print(f"📂 Found existing weights file: {weights_path}")
        weights_loaded = load_model_weights(model, weights_path)

    if weights_loaded:
        # Make predictions with loaded weights
        print("✅ Using existing model weights for predictions")
        train_predictions = model.predict(X_train_seq)
        test_predictions = model.predict(X_test_seq)
    else:
        # Train the model (either forced retrain or weights loading failed)
        if not retrain and os.path.exists(weights_path):
            print("🔄 Retraining model due to weights incompatibility...")
        elif retrain:
            print("🔄 Retraining model as requested...")
        else:
            print("🔄 Training new model (no weights file found)...")
            
        history = train_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=50, batch_size=32)

        # Save the model weights
        save_model_weights(model, weights_path)

        # Make predictions
        train_predictions = model.predict(X_train_seq)
        test_predictions = model.predict(X_test_seq)

    # Create a DataFrame with historical predictions and actual values
    combined_results_df = create_results_df(scaler_y, y_train_seq, y_test_seq, train_predictions, test_predictions,
                                            train_dates, test_dates)

    # Optionally, plot predictions vs actual data
    # Example: plot_predictions(combined_results_df, start_date='2024-10-14', end_date='2024-10-29', day=5)

    # Predict the future values based on the most recent 10 observations
    future_predictions_df = predict_for_most_recent_date(model, X, scaler_X, scaler_y, market_data,
                                                         window_size=window_size, prediction_days=prediction_days)

    # Plot the past 20 days and next 10 days of close prices, including investment signals
    combined_past_future_df = plot_past_and_future(combined_results_df, market_data, future_predictions_df,
                                                   window_size=20, signal_df=None)  # signal_df will be added later

    # Save combined past and future predictions to Excel
    output_file_path = data_lvc_path.replace('.xlsx', '_past_and_future_predictions.xlsx')
    combined_past_future_df.to_excel(output_file_path, index=False)
    print(f"Saved combined past and future predictions to '{output_file_path}'.")

    # Flag investment signal based on the latest predictions
    # Get the last actual close price from the market data
    last_actual_close_price = market_data['Close'].iloc[-1]
    # Flag if any predicted day has a close price >= last_actual_close_price + threshold
    signal_df, has_signal = flag_investment_signal(future_predictions_df, last_actual_close_price, threshold=threshold)

    # Save the signal to the Excel file in a separate sheet if signals are detected
    if has_signal:
        # Append the signal_df to the existing Excel file under 'Investment Signals' sheet
        with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            signal_df.to_excel(writer, sheet_name='Investment Signals', index=False)
        print(f"Saved investment signals to 'Investment Signals' sheet in '{output_file_path}'.")
    else:
        print("No investment signals to save.")

    # Plot past and future with investment signals
    combined_past_future_df = plot_past_and_future(combined_results_df, market_data, future_predictions_df,
                                                       window_size=20, signal_df=signal_df if has_signal else None)




if __name__ == "__main__":
    main_script() 