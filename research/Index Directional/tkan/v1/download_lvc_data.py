import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_lvc_data(start_date='2020-01-01', end_date=None):
    """
    Downloads LVC (Amundi 2x CAC 40) historical data from Yahoo Finance
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
        DataFrame: Processed data with required columns
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data for LVC.PA (Paris exchange)
    ticker = "LVC.PA"
    
    try:
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        
        # Reset index to get Date as a column
        data.reset_index(inplace=True)
        
        # Rename columns to match your existing format
        data.rename(columns={
            'Date': 'Date',
            'Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Volume': 'Volume'
        }, inplace=True)
        
        # Calculate 15-day Simple Moving Average
        data['SMAVG (15)'] = data['Close'].rolling(window=15).mean()
        
        # Select only the columns you need
        final_data = data[['Date', 'Close', 'High', 'Low', 'Volume', 'SMAVG (15)']].copy()
        
        # Remove rows with NaN values (first 14 rows due to moving average)
        final_data = final_data.dropna()
        
        print(f"Downloaded {len(final_data)} records for {ticker}")
        print(f"Date range: {final_data['Date'].min()} to {final_data['Date'].max()}")
        
        return final_data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def save_to_excel(data, file_path):
    """
    Saves the data to Excel file
    
    Args:
        data (DataFrame): The data to save
        file_path (str): Path to save the Excel file
    """
    try:
        data.to_excel(file_path, index=False)
        print(f"Data saved to: {file_path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "Data", "LVC_daily_updated.xlsx")
    
    # Download data from 2020 to present
    lvc_data = download_lvc_data(start_date='2020-01-01')
    
    if lvc_data is not None:
        # Display first few rows
        print("\nFirst 5 rows:")
        print(lvc_data.head())
        
        print("\nLast 5 rows:")
        print(lvc_data.tail())
        
        print(f"\nColumns: {lvc_data.columns.tolist()}")
        
        # Save to Excel
        save_to_excel(lvc_data, output_path)
        
        print(f"\nData shape: {lvc_data.shape}")
    else:
        print("Failed to download data")
