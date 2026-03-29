import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import time
import os

def scrape_yahoo_finance_history(ticker="LVC.PA", max_retries=3):
    """
    Scrapes historical data from Yahoo Finance history page.
    
    Args:
        ticker (str): Yahoo Finance ticker symbol
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        DataFrame: Historical data with Date, Open, High, Low, Close, Volume
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/history/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"🌐 Attempting to scrape Yahoo Finance history page (attempt {attempt + 1}/{max_retries})...")
            print(f"🔗 URL: {url}")
            
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            print(f"✅ Successfully retrieved page (status: {response.status_code})")
            print(f"📄 Page content length: {len(response.text)} characters")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the historical data table
            tables = soup.find_all('table')
            print(f"🔍 Found {len(tables)} tables on the page")
            
            if not tables:
                print("❌ No tables found on the page")
                continue
            
            # Find the table with historical data (usually the first or largest table)
            data_table = None
            for i, table in enumerate(tables):
                rows = table.find_all('tr')
                if len(rows) > 10:  # Historical data table should have many rows
                    data_table = table
                    print(f"📊 Using table {i+1} with {len(rows)} rows")
                    break
            
            if not data_table:
                print("❌ Could not find historical data table")
                continue
            
            # Extract data from table
            rows = data_table.find_all('tr')
            data = []
            
            # Debug: Check header row to understand column structure
            if len(rows) > 0:
                header_cells = rows[0].find_all(['td', 'th'])
                header_texts = [cell.get_text(strip=True) for cell in header_cells]
                print(f"📋 Table headers: {header_texts}")
            
            for i, row in enumerate(rows[1:]):  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 6:  # Date, Open, High, Low, Close, Adj Close, Volume
                    try:
                        date_text = cells[0].get_text(strip=True)
                        open_price = cells[1].get_text(strip=True)
                        high_price = cells[2].get_text(strip=True)
                        low_price = cells[3].get_text(strip=True)
                        # Use Close (cells[4]) as requested
                        close_price = cells[4].get_text(strip=True)
                        volume = cells[6].get_text(strip=True) if len(cells) > 6 else "0"
                        
                        # Clean and convert data
                        date_obj = datetime.strptime(date_text, "%b %d, %Y")
                        
                        # Clean price data (remove commas, handle dashes)
                        def clean_price(price_str):
                            if price_str == '-' or price_str == 'null':
                                return None
                            return float(price_str.replace(',', ''))
                        
                        def clean_volume(vol_str):
                            if vol_str == '-' or vol_str == 'null':
                                return 0
                            # Handle volume abbreviations (M, K, etc.)
                            vol_str = vol_str.replace(',', '')
                            if 'M' in vol_str:
                                return int(float(vol_str.replace('M', '')) * 1000000)
                            elif 'K' in vol_str:
                                return int(float(vol_str.replace('K', '')) * 1000)
                            else:
                                return int(vol_str) if vol_str.isdigit() else 0
                        
                        row_data = {
                            'Date': date_obj,
                            'Open': clean_price(open_price),
                            'High': clean_price(high_price),
                            'Low': clean_price(low_price),
                            'Close': clean_price(close_price),
                            'Volume': clean_volume(volume)
                        }
                        
                        data.append(row_data)
                        
                    except Exception as e:
                        print(f"⚠️ Error parsing row {i+1}: {e}")
                        continue
            
            if data:
                df = pd.DataFrame(data)
                df = df.sort_values('Date').reset_index(drop=True)
                print(f"✅ Successfully scraped {len(df)} records")
                print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
                return df
            else:
                print("❌ No valid data extracted from table")
                
        except requests.exceptions.RequestException as e:
            print(f"🌐 Network error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 5 seconds before retry...")
                time.sleep(5)
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 5 seconds before retry...")
                time.sleep(5)
    
    print(f"❌ Failed to scrape data after {max_retries} attempts")
    return pd.DataFrame()


def update_lvc_with_scraped_data(data_file_path, ticker="LVC.PA"):
    """
    Updates the LVC Excel file with scraped Yahoo Finance data.
    
    Args:
        data_file_path (str): Path to the existing Excel file
        ticker (str): Yahoo Finance ticker symbol
        
    Returns:
        str: Path to the updated file
    """
    print(f"🔄 Updating LVC data by scraping Yahoo Finance...")
    
    try:
        # Load existing data
        if not os.path.exists(data_file_path):
            print("❌ No existing data found!")
            return data_file_path
            
        print(f"📂 Loading existing data from: {data_file_path}")
        existing_data = pd.read_excel(data_file_path)
        
        # Store original structure
        original_columns = existing_data.columns.tolist()
        print(f"📋 Original columns: {original_columns}")
        
        existing_data['Date'] = pd.to_datetime(existing_data['Date'])
        last_date = existing_data['Date'].max()
        
        # Check original sort order
        is_descending = existing_data['Date'].iloc[0] > existing_data['Date'].iloc[-1]
        print(f"📅 Last date in file: {last_date.date()}")
        print(f"🔽 Sort order: {'DESCENDING' if is_descending else 'ASCENDING'}")
        
        # Scrape new data
        scraped_data = scrape_yahoo_finance_history(ticker)
        
        if scraped_data.empty:
            print("❌ No data scraped - keeping original file")
            return data_file_path
        
        # Filter for new dates only
        existing_dates = set(existing_data['Date'].dt.date)
        new_dates_only = scraped_data[~scraped_data['Date'].dt.date.isin(existing_dates)]
        
        if len(new_dates_only) == 0:
            print("✅ No new dates to add - data is up to date")
            return data_file_path
        
        print(f"🆕 Found {len(new_dates_only)} new dates to add")
        
        # Map scraped data to existing column structure
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
                    new_record[col] = None  # Will be calculated
                else:
                    new_record[col] = None
            new_records.append(new_record)
        
        # Combine data
        if new_records:
            new_df = pd.DataFrame(new_records)
            combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        else:
            combined_data = existing_data.copy()
        
        # Sort for calculation
        combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Recalculate SMAVG (15)
        if 'SMAVG (15)' in combined_data.columns:
            print("🔢 Recalculating SMAVG (15)...")
            combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
        
        # Restore original sort order
        if is_descending:
            combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
        
        # Preserve column order
        combined_data = combined_data[original_columns]
        
        # Save updated file
        combined_data.to_excel(data_file_path, index=False)
        
        print(f"✅ Updated data saved to: {data_file_path}")
        print(f"📊 Total records: {len(combined_data)}")
        print(f"📅 Date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
        print(f"🆕 New records added: {len(new_records)}")
        
        return data_file_path
        
    except Exception as e:
        print(f"❌ Error updating data: {e}")
        return data_file_path


if __name__ == "__main__":
    # Test the scraper
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Data", "LVC_daily.xlsx")
    result = update_lvc_with_scraped_data(data_file, "LVC.PA")
    print(f"Final result: {result}")

def scrape_yahoo_finance_history(ticker, start_date=None, end_date=None):
    """
    Scrapes historical data from Yahoo Finance history page
    
    Args:
        ticker (str): Yahoo Finance ticker symbol (e.g., 'LVC.PA')
        start_date (str): Start date in 'YYYY-MM-DD' format (optional)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
    
    Returns:
        DataFrame: Historical data with Date, Open, High, Low, Close, Volume columns
    """
    print(f"🕷️  Scraping Yahoo Finance history for {ticker}...")
    
    try:
        # Construct URL
        url = f"https://finance.yahoo.com/quote/{ticker}/history/"
        
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Make request
        print(f"📡 Fetching data from: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the historical data table
        # Look for table with historical prices
        table = soup.find('table', class_='table')
        if not table:
            # Try alternative selectors
            tables = soup.find_all('table')
            if tables:
                table = tables[0]  # Take first table found
            else:
                print("❌ Could not find data table on page")
                return pd.DataFrame()
        
        print("✅ Found data table")
        
        # Extract table rows
        rows = table.find_all('tr')
        if len(rows) < 2:
            print("❌ No data rows found in table")
            return pd.DataFrame()
        
        # Get headers from first row
        headers_row = rows[0]
        headers = [th.get_text(strip=True) for th in headers_row.find_all(['th', 'td'])]
        print(f"📋 Table headers: {headers}")
        
        # Extract data rows
        data_rows = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 6:  # Ensure we have enough columns
                row_data = [cell.get_text(strip=True) for cell in cells]
                data_rows.append(row_data)
        
        if not data_rows:
            print("❌ No data rows extracted")
            return pd.DataFrame()
        
        print(f"📊 Extracted {len(data_rows)} data rows")
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0])])
        
        # Clean and process the data
        df = clean_yahoo_finance_data(df)
        
        # Filter by date range if specified
        if start_date or end_date:
            df = filter_by_date_range(df, start_date, end_date)
        
        return df
        
    except Exception as e:
        print(f"❌ Error scraping Yahoo Finance: {e}")
        return pd.DataFrame()


def clean_yahoo_finance_data(df):
    """
    Cleans and standardizes the scraped Yahoo Finance data
    
    Args:
        df (DataFrame): Raw scraped data
        
    Returns:
        DataFrame: Cleaned data
    """
    print("🧹 Cleaning scraped data...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Try to identify date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or col == df.columns[0]:
            date_col = col
            break
    
    if date_col is None:
        print("❌ Could not identify date column")
        return df
    
    # Rename columns to standard format - handle Yahoo Finance column descriptions
    column_mapping = {}
    for i, col in enumerate(df.columns):
        col_lower = col.lower()
        if i == 0 or 'date' in col_lower:
            column_mapping[col] = 'Date'
        elif 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
        elif col_lower.startswith('close'):
            # Handle 'CloseClose price adjusted for splits.' format - this is the regular Close, not Adj Close
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'Volume'
        elif col_lower.startswith('adj close'):
            # Handle 'Adj CloseAdjusted close price...' format
            column_mapping[col] = 'Adj Close'
    
    df = df.rename(columns=column_mapping)
    print(f"📋 Renamed columns: {list(df.columns)}")
    
    # Keep only the columns we need
    required_cols = ['Date', 'Close', 'High', 'Low', 'Volume']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if 'Date' not in available_cols:
        print("❌ Date column not found after cleaning")
        return df
    
    df = df[available_cols]
    
    # Clean the data
    # Remove rows with missing dates or invalid data
    df = df[df['Date'].notna()]
    df = df[df['Date'] != '']
    
    # Convert date column
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].notna()]
    except Exception as e:
        print(f"⚠️  Error converting dates: {e}")
        return df
    
    # Clean numeric columns
    numeric_cols = ['Close', 'High', 'Low', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas and convert to numeric
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid numeric data
    df = df.dropna()
    
    # Sort by date (most recent first, like Yahoo Finance)
    df = df.sort_values('Date', ascending=False).reset_index(drop=True)
    
    print(f"✅ Cleaned data: {len(df)} rows")
    return df


def filter_by_date_range(df, start_date=None, end_date=None):
    """
    Filters DataFrame by date range
    
    Args:
        df (DataFrame): Data with Date column
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame: Filtered data
    """
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['Date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['Date'] <= end_date]
    
    return df


def update_lvc_with_scraped_data(data_file_path, ticker='LVC.PA'):
    """
    Updates LVC Excel file with scraped Yahoo Finance data
    
    Args:
        data_file_path (str): Path to LVC Excel file
        ticker (str): Yahoo Finance ticker
        
    Returns:
        str: Path to updated file
    """
    print(f"🔄 Updating LVC file with scraped data...")
    
    try:
        # Load existing data
        if not os.path.exists(data_file_path):
            print("❌ LVC file not found")
            return data_file_path
            
        existing_data = pd.read_excel(data_file_path)
        print(f"📂 Loaded existing data: {len(existing_data)} rows")
        
        # Get the date range we need
        existing_data['Date'] = pd.to_datetime(existing_data['Date'])
        last_date = existing_data['Date'].max()
        print(f"📅 Last date in file: {last_date.date()}")
        
        # Scrape new data
        scraped_data = scrape_yahoo_finance_history(ticker)
        
        if scraped_data.empty:
            print("❌ No data scraped")
            return data_file_path
        
        print(f"📥 Scraped {len(scraped_data)} rows")
        print(f"📅 Scraped date range: {scraped_data['Date'].min().date()} to {scraped_data['Date'].max().date()}")
        
        # Find new dates only
        existing_dates = set(existing_data['Date'].dt.date)
        new_data = scraped_data[~scraped_data['Date'].dt.date.isin(existing_dates)]
        
        if len(new_data) == 0:
            print("✅ No new dates to add")
            return data_file_path
        
        print(f"🆕 Found {len(new_data)} new dates")
        
        # Prepare new records to match existing structure
        original_columns = existing_data.columns.tolist()
        
        new_records = []
        for _, row in new_data.iterrows():
            new_record = {}
            for col in original_columns:
                if col == 'Date':
                    new_record[col] = row['Date']
                elif col == 'Close':
                    new_record[col] = row.get('Close', None)
                elif col == 'High':
                    new_record[col] = row.get('High', None)
                elif col == 'Low':
                    new_record[col] = row.get('Low', None)
                elif col == 'Volume':
                    new_record[col] = row.get('Volume', None)
                elif col == 'SMAVG (15)':
                    new_record[col] = None  # Will calculate later
                else:
                    new_record[col] = None
            new_records.append(new_record)
        
        # Combine data
        if new_records:
            new_df = pd.DataFrame(new_records)
            combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        else:
            combined_data = existing_data.copy()
        
        # Sort ascending for SMAVG calculation
        combined_data = combined_data.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Recalculate SMAVG (15)
        if 'SMAVG (15)' in combined_data.columns:
            print("🔢 Recalculating SMAVG (15)...")
            combined_data['SMAVG (15)'] = combined_data['Close'].rolling(window=15, min_periods=1).mean()
        
        # Restore original sort order (check if original was descending)
        is_descending = existing_data['Date'].iloc[0] > existing_data['Date'].iloc[-1]
        if is_descending:
            combined_data = combined_data.sort_values('Date', ascending=False).reset_index(drop=True)
            print("🔽 Restored DESCENDING sort order")
        
        # Ensure column order matches original
        combined_data = combined_data[original_columns]
        
        # Save updated data
        combined_data.to_excel(data_file_path, index=False)
        
        print(f"✅ Updated file saved: {data_file_path}")
        print(f"📊 Total records: {len(combined_data)}")
        print(f"🆕 New records added: {len(new_records)}")
        
        return data_file_path
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return data_file_path


if __name__ == "__main__":
    # Test the scraper
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Data", "LVC_daily.xlsx")
    update_lvc_with_scraped_data(data_file, 'LVC.PA')
