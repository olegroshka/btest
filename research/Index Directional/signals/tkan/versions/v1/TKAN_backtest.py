import os
import tensorflow as tf
from tkan import TKAN  # Ensure TKAN is correctly installed and compatible with your TensorFlow version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error, mean_squared_error


def process_stock(data_lvc_path, backtest_start_date='2022-01-01', window_size=10, prediction_days=10):
    """
    Processes the stock data, trains the TKAN model, and prepares data for backtesting.

    Args:
        data_lvc_path (str): Path to the Excel data file.
        backtest_start_date (str): Start date for the backtest in 'YYYY-MM-DD' format.
        window_size (int): Number of past days to consider for predictions.
        prediction_days (int): Number of days ahead to predict.

    Returns:
        model: Trained TensorFlow model.
        scaler_X: Scaler fitted on training features.
        scaler_y: Scaler fitted on training target.
        X_train: Training features DataFrame.
        market_data_train: Training market data DataFrame.
        X_backtest: Backtest features DataFrame.
        market_data_backtest: Backtest market data DataFrame.
        history: Training history object.
    """
    # Load the data
    market_data = pd.read_excel(data_lvc_path)

    # Convert 'Date' to datetime and sort in ascending order
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.sort_values('Date')
    market_data.set_index('Date', inplace=True)

    # Create lagged features
    market_data['Prior Close Price'] = market_data['Close'].shift(1)
    market_data['Prior Volume'] = market_data['Volume'].shift(1)
    market_data['Prior High'] = market_data['High'].shift(1)
    market_data['Prior Low'] = market_data['Low'].shift(1)
    market_data['Prior SMAVG (15)'] = market_data['SMAVG (15)'].shift(1)

    # Handle missing values by dropping rows with NaNs
    market_data = market_data.dropna()

    # Split data into training and backtest sets
    backtest_start_date = pd.to_datetime(backtest_start_date)
    market_data_train = market_data[market_data.index < backtest_start_date]
    market_data_backtest = market_data[market_data.index >= backtest_start_date]

    # Features and target
    X_train = market_data_train[['Prior Close Price', 'Prior Volume', 'Prior High', 'Prior Low', 'Prior SMAVG (15)']]
    y_train = market_data_train['Close']

    X_backtest = market_data_backtest[
        ['Prior Close Price', 'Prior Volume', 'Prior High', 'Prior Low', 'Prior SMAVG (15)']]
    y_backtest = market_data_backtest['Close']

    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers on training data and transform
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # Create sequences for training
    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i:i + window_size])  # Predict next 'window_size' days
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)

    # Define the TKAN model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=X_train_seq.shape[1:]),  # (window_size, 5)

        # TKAN Layers
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 1
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 2
        TKAN(100, return_sequences=True, use_bias=True),  # Layer 3

        tf.keras.layers.Dense(1)  # Output layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    return model, scaler_X, scaler_y, X_train, market_data_train, X_backtest, market_data_backtest, history


def perform_backtest(model, scaler_X, scaler_y, X_backtest, market_data_backtest, window_size=10, prediction_days=10,
                     initial_capital=100000):
    """
    Performs backtesting based on the specified investment criteria with portfolio tracking.

    Args:
        model: Trained TensorFlow model.
        scaler_X: Scaler fitted on training features.
        scaler_y: Scaler fitted on training target.
        X_backtest: Backtest features DataFrame.
        market_data_backtest: Backtest market data DataFrame.
        window_size (int): Number of past days to consider for predictions.
        prediction_days (int): Number of days ahead to predict.
        initial_capital (float): Starting capital in GBP.

    Returns:
        trades (list): List of dictionaries containing trade details.
        portfolio_history (DataFrame): Daily portfolio value tracking.
    """
    trades = []
    portfolio_history = []
    holding = False
    buy_price = 0
    buy_date = None
    sell_date = None
    sell_price = 0
    shares = 0
    capital = initial_capital

    backtest_dates = market_data_backtest.index
    total_days = len(backtest_dates)

    for current_idx in range(window_size, total_days):
        current_date = backtest_dates[current_idx]
        current_price = market_data_backtest.iloc[current_idx]['Close']

        # Record portfolio value at the start of the day
        portfolio_value = capital + shares * current_price
        portfolio_history.append({
            'Date': current_date,
            'Portfolio Value': portfolio_value
        })

        if not holding:
            # Extract the window_size days before the current day
            X_window = X_backtest.iloc[current_idx - window_size:current_idx]
            X_window_scaled = scaler_X.transform(X_window)
            X_window_scaled = X_window_scaled.reshape(1, window_size, -1)  # Shape: (1, window_size, 5)

            # Make predictions for the next 'prediction_days' days
            predictions_scaled = model.predict(X_window_scaled).flatten()
            predictions_actual = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

            # Determine if any prediction meets the buy criteria (price increases by at least 1 point)
            target = current_price + 1  # At least 1 point higher
            if np.any(predictions_actual >= target):
                # Find the first day in predictions that meets or exceeds the target
                days_ahead = np.argmax(predictions_actual >= target) + 1  # +1 because days_ahead starts at 1
                predicted_date = current_date + BDay(days_ahead)

                # Check if predicted_date exists in backtest data
                if predicted_date in backtest_dates:
                    actual_future_price = market_data_backtest.loc[predicted_date, 'Close']

                    if actual_future_price >= target:
                        # Execute the trade: Buy as many shares as possible
                        shares = capital // current_price  # Integer division to get whole shares
                        if shares > 0:
                            buy_price = current_price
                            buy_date = current_date
                            capital -= shares * buy_price

                            trades.append({
                                'Buy Date': buy_date,
                                'Buy Price': buy_price,
                                'Sell Date': predicted_date,
                                'Sell Price': actual_future_price,
                                'Shares': shares,
                                'Profit': (actual_future_price - buy_price) * shares
                            })

                            # Update portfolio value after buying
                            portfolio_value_after_buy = capital + shares * current_price
                            portfolio_history[-1]['Portfolio Value'] = portfolio_value_after_buy

                            # Update holding status
                            holding = True
                            sell_date = predicted_date
        else:
            # Currently holding; check if it's time to sell
            if current_date == sell_date:
                # Sell all shares at the actual_future_price
                sell_price = market_data_backtest.iloc[current_idx]['Close']
                capital += shares * sell_price
                profit = (sell_price - buy_price) * shares

                # Update the last trade with actual sell price and profit
                trades[-1]['Sell Price'] = sell_price
                trades[-1]['Profit'] = profit

                # Reset holdings
                shares = 0
                holding = False
                sell_date = None
                sell_price = 0
                buy_price = 0
                buy_date = None

                # Update portfolio value after selling
                portfolio_value = capital
                portfolio_history[-1]['Portfolio Value'] = portfolio_value

    # After the loop, if still holding, sell at the last available price
    if holding:
        final_date = backtest_dates[-1]
        final_price = market_data_backtest.iloc[-1]['Close']
        capital += shares * final_price
        profit = (final_price - buy_price) * shares

        trades[-1]['Sell Date'] = final_date
        trades[-1]['Sell Price'] = final_price
        trades[-1]['Profit'] = profit

        portfolio_value = capital
        portfolio_history.append({
            'Date': final_date,
            'Portfolio Value': portfolio_value
        })

        shares = 0
        holding = False
        sell_date = None
        sell_price = 0
        buy_price = 0
        buy_date = None

    # Convert portfolio_history to DataFrame
    portfolio_history_df = pd.DataFrame(portfolio_history)

    return trades, portfolio_history_df


def calculate_performance(trades, initial_capital=100000):
    """
    Calculates and prints performance metrics based on executed trades and portfolio growth.

    Args:
        trades (list): List of dictionaries containing trade details.
        initial_capital (float): Starting capital in GBP.

    Returns:
        performance (dict): Dictionary containing performance metrics.
    """
    if not trades:
        print("No trades were executed during the backtest period.")
        return {}

    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    total_profit = df_trades['Profit'].sum()
    average_profit = df_trades['Profit'].mean()
    winning_trades = df_trades[df_trades['Profit'] > 0]
    winning_ratio = len(winning_trades) / total_trades * 100
    max_profit = df_trades['Profit'].max()
    max_loss = df_trades['Profit'].min()

    # Calculate return on investment (ROI)
    roi = (total_profit / initial_capital) * 100

    print(f"\nBacktest Performance:")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Total Profit: £{total_profit:.2f}")
    print(f"Average Profit per Trade: £{average_profit:.2f}")
    print(f"Winning Trades: {len(winning_trades)} ({winning_ratio:.2f}%)")
    print(f"Maximum Profit in a Single Trade: £{max_profit:.2f}")
    print(f"Maximum Loss in a Single Trade: £{max_loss:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

    performance = {
        'Total Trades': total_trades,
        'Total Profit': total_profit,
        'Average Profit': average_profit,
        'Winning Trades': len(winning_trades),
        'Winning Ratio (%)': winning_ratio,
        'Max Profit': max_profit,
        'Max Loss': max_loss,
        'ROI (%)': roi
    }

    return performance


def plot_backtest_trades(trades, market_data_backtest, portfolio_history):
    """
    Plots the buy and sell trades on the actual closing price chart and portfolio value over time.

    Args:
        trades (list): List of dictionaries containing trade details.
        market_data_backtest (DataFrame): Backtest market data DataFrame.
        portfolio_history (DataFrame): Daily portfolio value tracking.
    """
    if not trades:
        print("No trades to plot.")
        return

    df_trades = pd.DataFrame(trades)

    # Plot Buy and Sell Signals
    plt.figure(figsize=(14, 7))
    plt.plot(market_data_backtest.index, market_data_backtest['Close'], label='Actual Close Price', color='blue')

    # Plot buy and sell points
    plt.scatter(df_trades['Buy Date'], df_trades['Buy Price'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(df_trades['Sell Date'], df_trades['Sell Price'], marker='v', color='red', label='Sell Signal', s=100)

    plt.title('Buy and Sell Signals on Actual Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (£)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Portfolio Value Over Time
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_history['Date'], portfolio_history['Portfolio Value'], label='Portfolio Value', color='purple')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (£)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plots the training and validation loss and MAE over epochs.

    Args:
        history: Training history object.
    """
    # Plot Loss Over Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot MAE Over Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


def main_script():
    """
    Main function to execute the stock prediction, backtest, and performance evaluation.
    """
    # Define parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_lvc_path = os.path.join(script_dir, "Data", "LVC_daily.xlsx")
    backtest_start_date = '2020-07-01'
    window_size = 10
    prediction_days = 10
    initial_capital = 100000  # Starting with £100,000

    # Process stock data and train the model
    model, scaler_X, scaler_y, X_train, market_data_train, X_backtest, market_data_backtest, history = process_stock(
        data_lvc_path, backtest_start_date, window_size, prediction_days
    )

    # Perform backtest
    trades, portfolio_history = perform_backtest(
        model, scaler_X, scaler_y, X_backtest, market_data_backtest,
        window_size=window_size, prediction_days=prediction_days, initial_capital=initial_capital
    )

    # Calculate and display performance metrics
    performance = calculate_performance(trades, initial_capital=initial_capital)

    # Save trades and portfolio history to Excel with separate tabs
    if trades:
        df_trades = pd.DataFrame(trades)
        portfolio_history_df = portfolio_history.copy()

        # Define output path
        backtest_output_path = os.path.join(script_dir, "Data", "backtest_results.xlsx")

        # Save to Excel with separate sheets
        with pd.ExcelWriter(backtest_output_path) as writer:
            df_trades.to_excel(writer, sheet_name='Trade History', index=False)
            portfolio_history_df.to_excel(writer, sheet_name='Portfolio Value', index=False)

        print(f"Saved backtest results to '{backtest_output_path}'.")

    # Plot trades on actual close price chart and portfolio value
    plot_backtest_trades(trades, market_data_backtest, portfolio_history)

    # Plot training history
    plot_training_history(history)

    # Optionally, plot cumulative returns
    if not portfolio_history.empty:
        portfolio_history['Cumulative Return'] = portfolio_history['Portfolio Value'] / initial_capital - 1
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_history['Date'], portfolio_history['Cumulative Return'], label='Cumulative Return',
                 color='orange')
        plt.title('Cumulative Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main_script()