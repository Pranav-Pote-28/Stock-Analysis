import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model("D:/Stock Analysis/Stock Analysis Model.keras")

st.header("Stock Market Predictor")

stock = st.text_input("Enter Stock symbol", "GOOG")
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80) : len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader("Price vs Moving_Average_50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, 'g')
plt.plot(ma_50_days, 'r')
plt.show()
st.pyplot(fig1)


st.subheader("Price vs Moving_Average_50 vs MA100")
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, 'g')
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.show()
st.pyplot(fig2)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100 : i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)


predict_values = model.predict(x)

scale = 1/scaler.scale_

predict_values = predict_values * scale
y = y * scale

st.subheader("Original Closing Price vs Prdicted Price")

fig3 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label = 'Original prices')
plt.plot(predict_values, 'r', label = 'Predicted values')
plt.legend()

plt.xlabel("Time")
plt.ylabel("Price")

plt.show()
st.pyplot(fig3)

mape = np.mean(np.abs(y - predict_values.flatten())/ y) * 100
accuracy = 100 - mape

st.subheader("Prediction Accuracy")
st.write(f"Accuracy : {accuracy:.2f}%")