import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam

# Global Variables
TICKER_SYMBOL = 'GOOGL'
START_DATE = '2014-1-1'
END_DATE = '2023-1-1'
PREDICTION_DAYS = 60
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

def fetch_data(ticker_symbol):
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(period='1d', start=START_DATE, end=END_DATE)

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_sequences(data, days):
    sequences = np.zeros((data.shape[0] - days, days))
    next_day_close_values = np.zeros((data.shape[0] - days))
    for i in range(days, data.shape[0]):
        sequences[i - days] = data[i - days:i].T
        next_day_close_values[i - days] = data[i]
    return np.expand_dims(sequences, -1), next_day_close_values

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, mc])
    return model, history

def make_predictions(model, data):
    return model.predict(data)

def main():
    data = fetch_data(TICKER_SYMBOL)
    scaled_data, scaler = preprocess_data(data)
    x_train, y_train = create_sequences(scaled_data, PREDICTION_DAYS)
    model = create_model((x_train.shape[1], 1))
    model, history = train_model(model, x_train, y_train)
    predictions = make_predictions(model, x_train)
    predictions = scaler.inverse_transform(predictions)

    plt.plot(data['Close'].values, color="black", label=f"Actual {TICKER_SYMBOL} Price")
    plt.plot(np.arange(PREDICTION_DAYS, PREDICTION_DAYS + len(predictions)), predictions, color="red", label=f"Predicted {TICKER_SYMBOL} Price")
    plt.title(f"{TICKER_SYMBOL} Share Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f'{TICKER_SYMBOL} Share Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
