# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date, time
import numpy as np  # Add numpy for array operations
import yfinance as yf  # Add yfinance for fetching stock data

# Install required libraries
try:
    from yahoofinancials import YahooFinancials
    import pandas_ta as ta
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
except ImportError as e:
    print(f"Error importing required libraries: {e}")

# Fetch historical stock data
df = yf.download('TSLA', start='2000-01-01', end=date.today(), progress=False)

# Preprocess data
df.dropna(inplace=True)

# Add technical indicators
df['RSI(2)'] = ta.rsi(df['Close'], length=2)
df['RSI(7)'] = ta.rsi(df['Close'], length=7)
df['RSI(14)'] = ta.rsi(df['Close'], length=14)
df['CCI(30)'] = ta.cci(close=df['Close'], length=30, high=df['High'], low=df['Low'])
df['CCI(50)'] = ta.cci(close=df['Close'], length=50, high=df['High'], low=df['Low'])
df['CCI(100)'] = ta.cci(close=df['Close'], length=100, high=df['High'], low=df['Low'])

# Prepare data labels
df['LABEL'] = np.where(df['Open'].shift(-2).gt(df['Open'].shift(-1)), "1", "0")
df.dropna(inplace=True)

# Split data into train and test sets
X = df[df.columns[6:-1]].values
y = df['LABEL'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Train neural network model
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)

# Evaluate model
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

# Visualize training and testing accuracy
print('Train Data Accuracy:')
print(classification_report(y_train, predict_train))

print('Testing Data Accuracy:')
print(classification_report(y_test, predict_test))

# Backtesting the model
df['Prediction'] = np.append(predict_train, predict_test)
df['Strategy Returns'] = np.where(df['Prediction'].eq("1"), df['Open'].shift(-2) - df['Open'].shift(-1), 0)
df['Strategy Returns'] = df['Strategy Returns'].cumsum()

# Visualize strategy returns
df.plot(y='Strategy Returns', title='Strategy Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Strategy Returns')
plt.show()

# Make today's return forecast
prediction = df.iloc[-1]['Prediction']
if prediction == "1":
    print("Today's return forecast: UP")
else:
    print("Today's return forecast: DOWN")
