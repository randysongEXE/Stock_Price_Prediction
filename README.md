# Stock_Price_Prediction
Stock Price Predictor using LSTM
This project aims to predict stock prices using a machine learning model, specifically, a Long Short-Term Memory (LSTM) model. The model is trained on historical stock price data fetched using the Yahoo Finance API (yfinance). The implementation is in Python and uses libraries such as tensorflow, pandas, numpy, matplotlib and sklearn.


Prerequisites:
You will need to have the following packages installed:
- numpy
- pandas
- yfinance
- sklearn
- tensorflow
- matplotlib

Use the package manager pip to install any missing packages.
- pip install numpy pandas yfinance sklearn tensorflow matplotlib

Clone the repository and run the script as follows:
git clone <repo_link> 
cd <repo_name> 
python3 main.py
Replace <repo_link> with the URL of your GitHub repository and <repo_name> with the name of the directory created when you cloned the repository.

Functionality:
The script fetches historical data for a specific ticker symbol from Yahoo Finance, preprocesses the data, and uses it to train an LSTM model. The model then predicts the stock prices for the coming days. The actual and predicted prices are visualized using matplotlib.

Configuration:
You can configure the model and prediction by modifying the following global variables in main.py:

- TICKER_SYMBOL: the ticker symbol of the stock you want to predict.
- START_DATE: the start date for the historical data.
- END_DATE: the end date for the historical data.
- PREDICTION_DAYS: the number of days to use for each prediction.

Results
---------
The model was tested with the Google Stock data, or 'GOOGL'. Here is the generated model:
![image](https://github.com/randysongEXE/Stock_Price_Prediction/assets/127687854/a12cace8-907d-44d2-97bd-43deab43ec46)



