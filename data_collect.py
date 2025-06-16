from binance.client import Client
import pandas as pd
from datetime import datetime
import os

#asset lists here 
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

#  Helper Function to get correct time interval 
def get_api_interval(interval_str):
    intervals = {
        "1min": Client.KLINE_INTERVAL_1MINUTE,
        "5min": Client.KLINE_INTERVAL_5MINUTE,
        "1hr": Client.KLINE_INTERVAL_1HOUR,
        "12hr": Client.KLINE_INTERVAL_12HOUR,
        "1day": Client.KLINE_INTERVAL_1DAY
    }
    interval = intervals.get(interval_str)
    if interval is None:
        raise ValueError("Unsupported time interval: choose from '1min', '5min', '1hr', '12hr', '1day'")
    return interval

#  Function to Fetch and Save Data 
def fetch_and_save_data(symbols, start_date_str, end_date_str, output_folder, interval_str="1hr", api_key=None, api_secret=None):
    """
    fetch  OHLCV data for given symbols and saves it to a specified folder.

        symbols (list): list of trading symbols (e.g., ['BTCUSDT']).
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
        output_folder (str): The path to the folder where CSV files will be saved.
        interval_str (str): The candle interval (e.g., '1hr').
    """
    client = Client(api_key, api_secret)
    api_interval = get_api_interval(interval_str)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for symbol in symbols:
        print(f"\nFetching {interval_str} data for {symbol} from {start_date_str} to {end_date_str}...")
        try:
            klines = client.get_historical_klines(
                symbol,
                api_interval,
                start_date_str,
                end_str=end_date_str
            )

            if not klines:
                print(f"No data found for {symbol} in the given range.")
                continue

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert timestamp to datetime and process numeric columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only the essential columns for our models
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            print(f"Successfully fetched {len(df)} records for {symbol}.")

            # Construct a filename compatible with our training scripts
            filename = f"{symbol}_{interval_str}_data_{start_date_str}_to_{end_date_str}.csv"
            full_path = os.path.join(output_folder, filename)
            
            # Save to CSV with the 'timestamp' column, not as an index.
            df.to_csv(full_path, index=False)
            
            print(f"Data for {symbol} saved to [{full_path}]")

        except Exception as e:
            print(f"An error occurred while fetching data for {symbol}: {e}")

#  Main  Function 
def run_data_gathering_process():
    start_date = "2020-01-01"
    # If end_date is in the future, the API will fetch data up to the latest available.
    end_date = "2025-06-07" 
    base_folder = "historic_data"
    stable_folder = os.path.join(base_folder, "stable")
    volatile_folder = os.path.join(base_folder, "volatile")
    
    print("--- Starting Data Gathering Process ---")

    # Fetch and save data for STABLE assets
    print("\n" + "="*50)
    print("Processing STABLE assets...")
    print("="*50)
    fetch_and_save_data(STABLE_ASSETS, start_date, end_date, stable_folder, interval_str="1hr")

    # Fetch and save data for HIGH VOLATILITY assets
    print("\n" + "="*50)
    print("Processing HIGH VOLATILITY assets...")
    print("="*50)
    fetch_and_save_data(HIGH_VOLATILITY_ASSETS, start_date, end_date, volatile_folder, interval_str="5min")
    
    print("\n---  Data Gathering Process Complete ---")

# Main Process
if __name__ == '__main__':
    run_data_gathering_process()