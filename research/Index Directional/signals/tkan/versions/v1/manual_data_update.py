"""
Manual data update helper for LVC daily data.
Use this when Yahoo Finance API is limited.
"""

import pandas as pd
import os
from datetime import datetime

def add_manual_data_from_csv(existing_file_path, csv_file_path):
    """
    Manually add data from a CSV file downloaded from Investing.com
    
    Args:
        existing_file_path: Path to existing LVC_daily.xlsx
        csv_file_path: Path to CSV file from Investing.com
    """
    print("📂 Loading existing data...")
    existing_data = pd.read_excel(existing_file_path)
    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
    
    print("📋 Original columns:", existing_data.columns.tolist())
    print("📊 Original records:", len(existing_data))
    print("📅 Latest date:", existing_data['Date'].max().date())
    
    # Check sort order
    is_descending = existing_data['Date'].iloc[0] > existing_data['Date'].iloc[-1]
    print("🔽 Sort order:", 'DESCENDING' if is_descending else 'ASCENDING')
    
    print("\n📂 Loading CSV data...")
    # Load CSV (assuming Investing.com format)
    csv_data = pd.read_csv(csv_file_path)
    print("📋 CSV columns:", csv_data.columns.tolist())
    print("📊 CSV records:", len(csv_data))
    
    # Convert date column (adjust column name as needed)
    if 'Date' in csv_data.columns:
        csv_data['Date'] = pd.to_datetime(csv_data['Date'])
    elif 'date' in csv_data.columns:
        csv_data['Date'] = pd.to_datetime(csv_data['date'])
    else:
        print("❌ Could not find date column in CSV")
        return
    
    # Find new dates only
    existing_dates = set(existing_data['Date'].dt.date)
    new_dates_only = csv_data[~csv_data['Date'].dt.date.isin(existing_dates)]
    
    if len(new_dates_only) == 0:
        print("✅ No new dates to add")
        return
    
    print(f"🆕 Found {len(new_dates_only)} new dates to add")
    
    # Map CSV columns to existing format
    new_records = []
    for _, row in new_dates_only.iterrows():
        new_record = {
            'Date': row['Date'],
            'Close': row.get('Close', row.get('close', row.get('Close*', None))),
            'High': row.get('High', row.get('high', None)),
            'Low': row.get('Low', row.get('low', None)),
            'Volume': row.get('Volume', row.get('volume', row.get('Vol.', 0))),
            'SMAVG (15)': None  # Will calculate later
        }
        new_records.append(new_record)
    
    # Add new data
    new_df = pd.DataFrame(new_records)
    combined_data = pd.concat([existing_data, new_df], ignore_index=True)
    
    # Sort for calculation
    combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # Recalculate SMAVG (15)
    print("🔢 Recalculating SMAVG (15)...")
    combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
    
    # Restore original sort order
    if is_descending:
        combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
    
    # Save
    combined_data.to_excel(existing_file_path, index=False)
    
    print(f"✅ Updated data saved!")
    print(f"📊 Total records: {len(combined_data)}")
    print(f"📅 New date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
    print(f"🆕 Added {len(new_records)} new records")

def create_sample_entry(existing_file_path, date_str, close_price, high_price, low_price, volume):
    """
    Manually add a single data entry
    
    Args:
        existing_file_path: Path to LVC_daily.xlsx
        date_str: Date string in 'YYYY-MM-DD' format
        close_price: Closing price
        high_price: High price
        low_price: Low price
        volume: Volume
    """
    print(f"📂 Adding single entry for {date_str}...")
    
    existing_data = pd.read_excel(existing_file_path)
    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
    
    # Check if date already exists
    new_date = pd.to_datetime(date_str)
    if new_date.date() in existing_data['Date'].dt.date.values:
        print(f"⚠️  Date {date_str} already exists in data")
        return
    
    # Check sort order
    is_descending = existing_data['Date'].iloc[0] > existing_data['Date'].iloc[-1]
    
    # Create new record
    new_record = pd.DataFrame({
        'Date': [new_date],
        'Close': [close_price],
        'High': [high_price],
        'Low': [low_price],
        'Volume': [volume],
        'SMAVG (15)': [None]
    })
    
    # Add to existing data
    combined_data = pd.concat([existing_data, new_record], ignore_index=True)
    
    # Sort for calculation
    combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # Recalculate SMAVG (15)
    combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
    
    # Restore original sort order
    if is_descending:
        combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
    
    # Save
    combined_data.to_excel(existing_file_path, index=False)
    
    print(f"✅ Added entry for {date_str}")
    print(f"📊 Total records: {len(combined_data)}")

if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    existing_file = os.path.join(script_dir, "Data", "LVC_daily.xlsx")
    
    print("Manual Data Update Helper")
    print("1. To add from CSV: add_manual_data_from_csv(existing_file, 'path_to_csv')")
    print("2. To add single entry: create_sample_entry(existing_file, '2025-08-30', 45.67, 46.12, 45.23, 12345)")
    print("")
    print("Current file status:")
    
    if os.path.exists(existing_file):
        df = pd.read_excel(existing_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Records: {len(df)}")
        print(f"Latest date: {df['Date'].max().date()}")
        
        import datetime
        days_missing = (datetime.date.today() - df['Date'].max().date()).days
        print(f"Days missing: {days_missing}")
    else:
        print("File not found")
