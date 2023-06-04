# Stock_Price_Prediction
Stock Price Predictor using LSTM
This project aims to predict stock prices using a machine learning model, specifically, a Long Short-Term Memory (LSTM) model. The model is trained on historical stock price data fetched using the Yahoo Finance API (yfinance). The implementation is in Python and uses libraries such as tensorflow, pandas, numpy, matplotlib and sklearn.

Getting Started
Prerequisites
You will need to have the following packages installed:

numpy
pandas
yfinance
sklearn
tensorflow
matplotlib
Use the package manager pip to install any missing packages.

bash
Copy code
pip install numpy pandas yfinance sklearn tensorflow matplotlib
Usage
Clone the repository and run the script as follows:

bash
Copy code
git clone <repo_link>
cd <repo_name>
python3 main.py
Replace <repo_link> with the URL of your GitHub repository and <repo_name> with the name of the directory created when you cloned the repository.

Functionality
The script fetches historical data for a specific ticker symbol from Yahoo Finance, preprocesses the data, and uses it to train an LSTM model. The model then predicts the stock prices for the coming days. The actual and predicted prices are visualized using matplotlib.

Configuration
You can configure the model and prediction by modifying the following global variables in main.py:

TICKER_SYMBOL: the ticker symbol of the stock you want to predict.
START_DATE: the start date for the historical data.
END_DATE: the end date for the historical data.
PREDICTION_DAYS: the number of days to use for each prediction.
Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
MIT


