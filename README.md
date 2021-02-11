# LSTM-with-indicators
Changes that are made:
LSTM predicts rise and fall of stock with sma indicator for next day and not every month, see intialize section in main.py
LSTM is only trained at the begining of initialize with history of sma50, sm200 and close price 
Backtest only occurs for one month(Jan 11 to Feb 11), to ensure predictions from lstm are only made for one month after training. LSTM will have to be retrained with the previous testing data(Jan to feb), if backtesting for the next month(march).   
