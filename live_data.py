# To gather live data for the model to predict
import json 
import websocket # pip install websocket-client
import pandas as pd
import joblib # Use joblib for loading models
import numpy as np
import os # For path joining

# Define assets and construct the WebSocket stream URL
assets_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
kline_streams = [coin.lower() + '@kline_1h' for coin in assets_symbols]
socket_streams_param = '/'.join(kline_streams)
socket_url = "wss://stream.binance.com:9443/stream?streams=" + socket_streams_param

# Function to load models using joblib
def load_models(symbols):
    loaded_models = {}
    models_dir = "trained_models" # Assuming models are in this directory
    for symbol in symbols:
        model_path = os.path.join(models_dir, f"{symbol}_rfr_model.pkl")
        try:
            model = joblib.load(model_path)
            loaded_models[symbol] = model
            print(f"Model for {symbol} loaded successfully from {model_path}. Type: {type(model)}")
        except FileNotFoundError:
            print(f"Error: Model file not found for {symbol} at {model_path}")
            loaded_models[symbol] = None # Or handle as critical error
        except Exception as e:
            print(f"Error loading model for {symbol} from {model_path}: {e}")
            loaded_models[symbol] = None
    return loaded_models

# Load models once globally
print("Loading models...")
models_dict = load_models(assets_symbols)

# Function to process incoming kline data and make predictions
def process_kline_and_predict(kline_message_data):
    """
    Processes a single kline data object and makes a prediction if a model exists.
    This function expects data from a *closed* kline.
    """
    kline_info = kline_message_data['k']
    symbol = kline_info['s']   
    
    # Extract features from the closed kline
    # These are the features of the hour that just *closed*
    open_price = float(kline_info['o'])
    high_price = float(kline_info['h'])
    low_price = float(kline_info['l'])
    close_price = float(kline_info['c']) # This is the closing price of the completed hour
    volume = float(kline_info['v'])

    event_time = pd.to_datetime(kline_message_data['E'], unit='ms') # Timestamp of the event
    kline_close_time = pd.to_datetime(kline_info['T'], unit='ms') # Timestamp when kline actually closed

    # Prepare features for prediction (must match training features and order)
    # X = data[['open', 'high', 'low', 'close', 'volume']]
    features_for_prediction = np.array([[open_price, high_price, low_price, close_price, volume]])

    # Make prediction if model for the symbol is loaded
    if symbol in models_dict and models_dict[symbol] is not None:
        model = models_dict[symbol]
        try:
            # The model predicts the 'target_close' which is the next hour's closing price
            predicted_next_hour_close = model.predict(features_for_prediction)[0]
            print(f"[{event_time}] Prediction for {symbol} (based on kline closed at {kline_close_time}):")
            print(f"  Last Closed Hour's OHLCV: O:{open_price:.2f} H:{high_price:.2f} L:{low_price:.2f} C:{close_price:.2f} V:{volume:.2f}")
            print(f"  Predicted Close for NEXT HOUR: {predicted_next_hour_close:.2f}")
            
            if predicted_next_hour_close > close_price:
                print(f"  Signal: Potential UP trend for {symbol} in the next hour.")
            elif predicted_next_hour_close < close_price:
                print(f"  Signal: Potential DOWN trend for {symbol} in the next hour.")
            else:
                print(f"  Signal: Potential SIDEWAYS movement for {symbol} in the next hour.")

        except Exception as e:
            print(f"Error during prediction for {symbol}: {e}")
    else:
        print(f"Model for {symbol} not found or not loaded. Cannot predict.")

# WebSocket event handlers
def on_open(ws):
    print(f"[WebSocket Opened] Subscribed to streams: {socket_streams_param}")

def on_error(ws, error):
    print(f"[WebSocket Error] {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"[WebSocket Closed] Code: {close_status_code}, Message: {close_msg}")
    # Add reconnection logic here if needed

def on_message(ws, message_str):
    """
    Handles incoming messages from the WebSocket.
    Only processes kline data if the kline is marked as closed ('x': True).
    """
    try:
        # print(f"[Raw Message Received] {message_str[:250]}{'...' if len(message_str) > 250 else ''}") # For debugging
        message_data = json.loads(message_str)
        
        # Check if it's a kline event from one of our subscribed streams
        if 'stream' in message_data and '@kline_1h' in message_data['stream']:
            kline_payload = message_data['data'] # This is the actual kline data object
            
            # 'k' contains the kline/candlestick data
            # 'x' is a boolean: true if this kline is closed, false if not.
            if kline_payload['k']['x']: 
                print(f"\n--- Closed 1-Hour Kline Received for {kline_payload['k']['s']} at {pd.to_datetime(kline_payload['E'], unit='ms')} ---")
                process_kline_and_predict(kline_payload)
            # else:
                # This is an update for the currently open kline, not yet closed.
    except json.JSONDecodeError:
        print(f"[JSON Decode Error] Unparseable message: {message_str[:250]}")
    except Exception as e:
        print(f"[Error in on_message processing]: {e} - Message snippet: {message_str[:250]}")

# --- Main WebSocket connection ---
if __name__ == "__main__":
    print(f"Attempting to connect to WebSocket: {socket_url}")
    # Create dummy model files if they don't exist for testing live_data.py independently
    os.makedirs('trained_models', exist_ok=True)
    for sym in assets_symbols:
        model_file = os.path.join('trained_models', f"{sym}_rfr_model.pkl")
        if not os.path.exists(model_file):
            print(f"Creating dummy model for {sym} at {model_file} for testing live_data.py...")
            from sklearn.ensemble import RandomForestRegressor as RFR_dummy
            dummy_rfr_model = RFR_dummy(n_estimators=5)
            dummy_rfr_model.fit(np.random.rand(10, 5), np.random.rand(10)) # Fit with dummy data
            joblib.dump(dummy_rfr_model, model_file)

    # Re-load models in case dummy ones were created
    models_dict = load_models(assets_symbols)


    ws_app = websocket.WebSocketApp(socket_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
    ws_app.run_forever()






