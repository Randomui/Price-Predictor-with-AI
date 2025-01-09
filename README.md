# Price-Predictor-with-AI
# Cryptocurrency Price Predictor

This project predicts the future prices of a cryptocurrency (e.g., Bitcoin) using historical data and a Long Short-Term Memory (LSTM) neural network.

## Features

- Fetches historical OHLCV data from the Binance exchange.
- Prepares the data by calculating percentage changes, moving averages, and normalizing the data.
- Defines and trains an LSTM model using TensorFlow.
- Generates future price predictions for a specified number of weeks.
- Plots the actual historical prices and the predicted future prices.

## Requirements

- Python 3.6+
- `ccxt`
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `tensorflow`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Randomui/Price-Predictor-with-AI.git
    cd Price-Predictor-with-AI
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Fetch historical data:
    ```python
    df = fetch_historical_data('BTC/USDT')
    ```

2. Prepare the data:
    ```python
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
    ```

3. Train the model:
    ```python
    model = train_model(X_train, y_train)
    ```

4. Generate future predictions (104 weeks forecast):
    ```python
    future_predictions = generate_future_predictions(model, df, scaler, weeks=104)
    print(future_predictions.head())
    ```

5. Visualize the results:
    ```python
    plot_predictions(df, future_predictions)
    ```

## Example

To run the entire process, execute the script:
```sh
python ai_predictor.py
