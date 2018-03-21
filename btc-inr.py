# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# load json and create model
json_file = open('my_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''

# Importing the training set
dataset_train = pd.read_csv('BTC_INR_train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, 1744):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('BTC_INR_test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_stock_price = np.concatenate((training_set[0:1744], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(1744, 1775):
    inputs.append(scaled_real_stock_price[i-20:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#predicted_stock_price = sc.transform(predicted_stock_price)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real BTC-INR')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC-INR')
plt.title('BTC-INR Prediction')
plt.xlabel('Time')
plt.ylabel('BTC-INR Price')
plt.legend()
plt.show()

'''
from keras.models import load_model
from keras.models import model_from_json
#regressor.save('my_model.h5')
 
# serialize model to JSON
model_json = regressor.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
'''
