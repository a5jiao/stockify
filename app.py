# Import required libraries
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

# Load model
model = load_model('Stock Prediction Model.keras')

st.header('Stock Market Predictor')

# Ticker input
tick = st.text_input('Enter Stock Ticker', 'GOOG')
start = '2012-01-01'
end = datetime.now()

# Minimum required data points for model to work
min_required_days = 375

# Check if the ticker is valid
if not yf.Ticker(tick).history(period="1d").empty:
    # Ticker is valid, download stock history
    data = yf.download(tick, start=start, end=end)

    # Check if the downloaded data is sufficient
    if len(data) < min_required_days:
        st.error(f"Insufficient data for analysis. Required: {min_required_days}, Available: {len(data)}")
        st.stop()  # Terminate execution if not enough data
else:
    # Ticker is not valid, output message and terminate
    st.error("Invalid Stock Ticker. Please enter a valid ticker.")
    st.stop()  # Terminate execution

# Get the name of the stock
stock = yf.Ticker(tick)
stock_name = stock.info['longName']

st.subheader('Stock Data for ' + tick.upper() + " (" + stock_name + ")")
st.write(data)

# Remove null data points to clean data
data.dropna(inplace=True)

# Initialize 80% of data points for training data set
# 20% of data points for testing data set
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8): len(data)])

# Set scale for data
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data_train)

# Scaling data and obtaining last 100 days of data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)
data_train_scale = scaler.fit_transform(data_train)

# Display plots of Price vs MA50 vs MA100 vs MA200
st.subheader('Price vs 50 Day Moving Average')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price vs 50 Day MA vs 100 Day MA')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price vs 100 Day MA vs 200 Day MA')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Create training data set
x = []
y = []

# Iterate starting from 100 since our LSTM model is going to use 
#  100 previous time steps as input
for i in range(100, data_test_scale.shape[0]): 
    x.append(data_test_scale[i-100:i]) # Appending scaled prev 100 data pts
    y.append(data_test_scale[i,0]) # Appending next scaled data pt

# Convert to array
x,y = np.array(x), np.array(y)

# Predict prices using model
predict = model.predict(x)

# Scale values back to normal
predict = scaler.inverse_transform(predict.reshape(-1, 1))
y = scaler.inverse_transform(y.reshape(-1, 1))

# Display predicted prices
st.subheader('Original Price vs Predicted Price for ' + tick.upper())
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'b', label='Original Price')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()
st.pyplot(fig4)


# Select last 100 days from dataset
last_100_days = data_train_scale[-100:]  # Keep it in the original format for slicing

# Initialize a list to store the predictions
predicted_prices = []

# Predict the next 7 days
for _ in range(7):
    # Reshape the last_100_days to fit the model input shape
    x_test = np.array([last_100_days])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Predict the next day's price
    predicted_price = model.predict(x_test)
    
    # Inverse scaling the prediction
    predicted_price_unscaled = scaler.inverse_transform(predicted_price)
    
    # Append the predicted price to predicted_prices
    predicted_prices.append(predicted_price_unscaled[0][0])
    
    # Update the last_100_days for the next prediction
    last_100_days = np.append(last_100_days, predicted_price)[1:].reshape(-100, 1)  # Ensure it's reshaped back to (-100, 1)

# Plotting the predicted prices
st.subheader('Predicted Close Price ' + tick.upper() + ' for Next 7 Days')
fig5 = plt.figure(figsize=(10,6))
days = range(1, 8)
plt.plot(days, predicted_prices, 'r', label='Predicted Close Price')
plt.title('Predicted ' + tick + ' Prices for the Next 7 Days')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)

last_100_days_30 = data_train_scale[-100:]  # Keep it in the original format for slicing

# Initialize a list to store the scaled predictions for 30 days prediction
predicted_prices_30 = []

# Predict the next 30 days
for _ in range(30):
    # Reshape the last_100_days_30 to fit the model input shape
    x_test_30 = np.array([last_100_days_30])
    x_test_30 = np.reshape(x_test_30, (x_test.shape[0], x_test.shape[1], 1))
    
    # Predict the next day's price
    predicted_price_30 = model.predict(x_test_30)
    
    # Inverse scaling the prediction
    predicted_price_unscaled_30 = scaler.inverse_transform(predicted_price_30)
    
    # Append the predicted price to predicted_prices
    predicted_prices_30.append(predicted_price_unscaled_30[0][0])
    
    # Update the last_100_days_30 for the next prediction
    last_100_days_30 = np.append(last_100_days_30, predicted_price_30)[1:].reshape(-100, 1)  # Ensure it's reshaped back to (-100, 1)

# Plotting the predicted prices
st.subheader('Predicted Close Price ' + tick.upper() + ' for Next 30 Days')
fig6 = plt.figure(figsize=(10,6))
days_30 = range(1, 31)
plt.plot(days_30, predicted_prices_30, 'r', label='Predicted Close Price')
plt.title('Predicted ' + tick + ' Prices for the Next 30 Days')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig6)

df_30 = pd.DataFrame({'Day': days_30, 'Close': predicted_price_30})
display(df_30)

st.write("&copy; 2024 Alfred Jiao")
