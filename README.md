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
The script fetches historical data for a specific ticker symbol from Yahoo Finance, preprocesses the data, and uses it to train an LSTM model. The model then predicts the stock prices for the coming days. The actual and predicted prices are visualized using matplotlib. The final graph gives the time in trading days, and it is through the period 2014-2023. However, this can be changed at the user's discretion.

Configuration:
You can configure the model and prediction by modifying the following global variables in main.py:

- TICKER_SYMBOL: the ticker symbol of the stock you want to predict.
- START_DATE: the start date for the historical data.
- END_DATE: the end date for the historical data.
- PREDICTION_DAYS: the number of days to use for each prediction.

Results
---------
The 'app2.0' model was tested with the Google Stock data, or 'GOOGL'. Here is the generated model:
![image](https://github.com/randysongEXE/Stock_Price_Prediction/assets/127687854/a12cace8-907d-44d2-97bd-43deab43ec46)

The model is fairly accurate - however, it should not be used to make actual financial decisions, as it does not take into account risk management and is susceptible to overfitting (a problem in machine learning where the training data may not accurately reflect real-world results). 

Interestingly enough, the 'app' model, with negligable differences, has a slight overprediction as shown below:
![image](https://github.com/randysongEXE/Stock_Price_Prediction/assets/127687854/101c69b3-2cf4-4e68-9f3f-a89d271becb9)

This is likely due to package differences in the Python environment, where different versions of tensorflow may cause different outputs.

Here are the predictions for CocaCola, Microsoft, and Nvidia, respectively.

![image](https://github.com/randysongEXE/Stock_Price_Prediction/assets/127687854/fb36f1fb-49cc-455e-867f-5cb6f26b91a1)
![image](https://github.com/randysongEXE/Stock_Price_Prediction/assets/127687854/c36e9ffc-1636-4297-912c-76b8dfdbf012)






