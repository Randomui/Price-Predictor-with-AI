import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Fetch Historical Data
def fetch_historical_data(symbol, timeframe='1w', limit=1000):
    exchange = ccxt.binance({
        'apiKey': 'api-Key',#Replace with api-key
        'secret': 'secret',#Replace with  secret
    })
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    all_ohlcv = []
    while since < exchange.milliseconds():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Step 2: Prepare Data
def prepare_data(df):
    df['pct_change'] = df['close'].pct_change()
    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    df['target_price'] = df['close'].shift(-1)  # Predict next week's closing price
    df = df.dropna()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 'pct_change', 'ma_7', 'ma_30']])
    
    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 3])  # 'close' price is the target
    X, y = np.array(X), np.array(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

# Step 3: Define and Train LSTM Model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model

# Step 4: Generate Future Predictions
def generate_future_predictions(model, df, scaler, weeks=104):
    # Use the last 60 rows for the initial input
    future_data = df.iloc[-60:].copy()
    predictions = []
    
    # Calculate minimum historical price
    min_historical_price = df['close'].min()
    
    # Initialize the starting date
    last_date = df['timestamp'].iloc[-1]

    for week in range(weeks):
        # Prepare the input data
        scaled_data = scaler.transform(future_data[['open', 'high', 'low', 'close', 'volume', 'pct_change', 'ma_7', 'ma_30']])
        X_input = np.array([scaled_data[-60:]])

        # Predict next price
        next_price_scaled = model.predict(X_input)[0][0]
        next_price = scaler.inverse_transform([[0, 0, 0, next_price_scaled, 0, 0, 0, 0]])[0][3]

        # Add some randomness (e.g., volatility) to the price prediction
        volatility = np.random.normal(-0.02, 0.04)  # Centered around 0% with a wider range
        next_price += next_price * volatility

        # Allow price to go below historical minimum but within a reasonable range
        if next_price < 0.9 * min_historical_price:
            next_price = max(next_price, 0.9 * min_historical_price)

        # Create next row with updated values
        next_row = future_data.iloc[-1:].copy()
        next_row['open'] = next_row['close']
        next_row['close'] = next_price
        next_row['volume'] = next_row['volume'] * (1 + np.random.normal(0, 0.01))  # Add slight variability to volume
        next_row['timestamp'] = last_date + timedelta(days=7 * (week + 1))  # Increment date by one week

        # Append the new row to `future_data`
        future_data = pd.concat([future_data, next_row], ignore_index=True)

        # Update `pct_change` dynamically
        future_data['pct_change'].iloc[-1] = (next_price - next_row['open']) / next_row['open']

        # Update moving averages dynamically
        if len(future_data) >= 7:
            future_data['ma_7'].iloc[-1] = future_data['close'].rolling(7).mean().iloc[-1]
        if len(future_data) >= 30:
            future_data['ma_30'].iloc[-1] = future_data['close'].rolling(30).mean().iloc[-1]

        # Collect prediction data
        predictions.append({'date': next_row['timestamp'].values[0], 'price': next_price})

        # Debugging print statement
        print(f"Week {week + 1}: Predicted price = {next_price}")

    return pd.DataFrame(predictions)



# Step 5: Plot Predictions
def plot_predictions(df, predictions):
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="darkgrid")
    
    # Convert timestamp to date for weekly formatting
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # Plot actual prices
    sns.lineplot(x='timestamp', y='close', data=df, label='Actual Price', color='blue')
    
    # Plot predicted prices
    sns.lineplot(x='date', y='price', data=predictions, label='Predicted Price', color='orange', linestyle='--')
    
    # Customize date formatting
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))  # Show dates in "Month Year" format
    plt.xticks(rotation=45)
    
    # Add titles and labels
    plt.title("Price Predictions for the Next Two Years")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Fetch data
    df = fetch_historical_data('BTC/USDT')
    
    # Prepare data
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Generate future predictions (104 weeks forecast)
    future_predictions = generate_future_predictions(model, df, scaler, weeks=104)
    print(future_predictions.head())
    
    # Visualize results
    plot_predictions(df, future_predictions)
