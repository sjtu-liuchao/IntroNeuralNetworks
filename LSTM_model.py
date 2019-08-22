import pandas as pd
import numpy as np
import get_prices as hist
import tensorflow as tf
from preprocessing import DataProcessing
import pandas_datareader.data as pdr
import yfinance as fix
import matplotlib.pyplot as plt
fix.pdr_override()

start = "2003-01-01"
end = "2018-01-01"

hist.get_stock_data("AAPL", start_date=start, end_date=end)
process = DataProcessing("stock_prices.csv", 0.9)
process.gen_test(10)
process.gen_train(10)

X_train = process.X_train / np.array([200, 1e9])  # 归一化， 包括Adj Close 和 Volume
Y_train = process.Y_train / 200

X_test = process.X_test / np.array([200, 1e9])
Y_test = process.Y_test / 200

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(10, 2), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

h=model.fit(X_train, Y_train, epochs=50)

print("history.loss:",h.history)
plt.title('loss') 
losses=h.history['loss']
plt.plot(losses, label='loss')  
plt.legend()  
plt.show()

print(model.evaluate(X_test, Y_test))

X_predict = model.predict(X_test)
plt.title("Results") 
plt.plot(Y_test * 200, label="Actual", c="blue")
plt.plot(X_predict * 200, label="Predict", c="red")
plt.legend()
plt.show()
plt.savefig('stock.png')

data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
stock = data[["Adj Close", "Volume"]]
X1_predict = np.array(stock) / np.array([200, 1e9])
X1_predict = X1_predict.reshape(1, -1, 2)
print("predict:")
print(model.predict(X1_predict)*200)
# If instead of a full backtest, you just want to see how accurate the model is for a particular prediction, run this:
#data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
#stock = data["Adj Close"]
#X_predict = np.array(stock).reshape((1, 10)) / 200
#print("predict:")
#print(model.predict(X_predict)*200)
