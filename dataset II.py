# Ensure pandas_ta is installed
try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta as ta

# Your DataFrame operations
# Assuming you've already imported pandas and have a DataFrame named df

# Calculate RSI and CCI
df['RSI(2)'] = ta.rsi(df['Close'], length=2)
df['RSI(7)'] = ta.rsi(df['Close'], length=7)
df['RSI(14)'] = ta.rsi(df['Close'], length=14)
df['CCI(30)'] = ta.cci(close=df['Close'], length=30, high=df['High'], low=df['Low'])
df['CCI(50)'] = ta.cci(close=df['Close'], length=50, high=df['High'], low=df['Low'])
df['CCI(100)'] = ta.cci(close=df['Close'], length=100, high=df['High'], low=df['Low'])

# Drop NaN Values
df = df.dropna()

# Plotting
df[['RSI(2)', 'RSI(7)', 'RSI(14)', 'CCI(100)']].plot(figsize=(12, 6), title='RSI and CCI')

# Display the head of the DataFrame
print(df.head())
