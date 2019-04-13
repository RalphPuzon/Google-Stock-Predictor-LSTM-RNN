import os
os.chdir('C:\\Users\\Ralph\\Desktop\\Courses\\DeepLearning\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 3 - Recurrent Neural Networks (RNN)')

#-----------------------------------------------------Recurrent Neural Networks
"""we are tasked with using an LSTM RNN to predict google open stock prices for
the year, given the data for the first month of the year."""

#PREPROCESSING:
#Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the training data:
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
"""the 1:2 is a trick to create an array with one column(remember end-1)"""

#Feature Scaling:
#Normalization: best used on sigmoid RNNs, Norm = (x-min(x)/(max(x)-min(x)))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#number of timesteps: 
#X_train - input of NN, y_train - output of NN
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) #memory from points 0 - 59
    y_train.append(training_set_scaled[i, 0]) #memory from point 60
    
X_train, y_train = np.array(X_train), np.array(y_train) 
"""multiple assignment to convert Xtr and ytr into numpy arrays in one line"""

#adding an extra dimension via the reshape function:

#newshape params:
#keras -> recurrent layers = input shape for proper shaping for the RNN
#RNN shape in order -> (batch_size(rows), timesteps(cols), input_dim)

X_train = np.reshape(X_train, newshape = (X_train.shape[0],
                                          X_train.shape[1],
                                          1))

#BUILDING THE RNN:
#Libraries:
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#initialize RNN: we are now creating a regressor (numerical predictions)
regressor = Sequential()
#add the 1st layer + dropout:
regressor.add(LSTM(units = 55,
                   return_sequences=True,
                   input_shape = (X_train.shape[1],
                                  1)))
regressor.add(Dropout(0.2))

#2nd layer:
regressor.add(LSTM(units = 55,
                   return_sequences=True))
regressor.add(Dropout(0.2))

#3rd Layer:
regressor.add(LSTM(units = 55,
                   return_sequences=True))
regressor.add(Dropout(0.2))

#4th Layer:
regressor.add(LSTM(units = 55,
                   return_sequences=False)) #you may omit retseq, F is default
regressor.add(Dropout(0.2))

#output layer, regular ANN layer
regressor.add(Dense(units = 1)) #return is the pred stock price at i+1

#Compile:
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

#fitting RNN to dataset
regressor.fit(X_train, y_train, epochs = 110, batch_size = 32)

#PREDICTING AND VISUALIZING STOCK PRICES USING THE RNN:
#get the stock price of 2017 (the test set)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values


#predict sp2017
"""there is a scaling issue, do not simply apply the fit transform to the test
set, as it will change the values""" 

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']),
                          axis = 0) #contains train and test data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

"""dataset total contains both train and test data. the difference of lengths
of the total and test removes the indexes for the test. the additional - 60 is
for the 60 previous days needed for the input to predict the one test point.
inputs is then set as the range of dataset 60 days prior to the first test data
point. the values method call turns the dataframe to an array."""
#--reshaping into a 3d numpy array as the NN expects this format:
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#--recreating the data structure from preprocessing to apply to the prediction:

X_test = []
for i in range(60, 80): #there are only 20 financial days to predict.
    X_test.append(inputs[i-60:i, 0]) 
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, newshape = (X_test.shape[0],
                                          X_test.shape[1],
                                          1))
#PREDICTION:
predicted_stock_price = regressor.predict(X_test)
#remove scaling:
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#viz
plt.plot(real_stock_price, color = 'red',
         label = 'Real Google Stock Price (Jan 2017)')

plt.plot(predicted_stock_price, color = 'blue',
         label = 'Predicted Google Stock Price  (Jan 2017)')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time(Jan 03 - Jan 31)')
plt.ylabel('Price')
plt.legend()
plt.show()

#Evaluate true value via Root Mean Squared Error(RMSE);
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
relative_error = rmse/(max(real_stock_price) - min(real_stock_price))


"""
#GRID SEARCH:
#gs does not accept keras as pnp, you need to wrap your function,
#see code below

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(LSTM(units = 50,
                       return_sequences=True,
                       input_shape = (X_train.shape[1],
                                      1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50,
                       return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50,
                       return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1)) 
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return regressor

regressor = KerasRegressor(build_fn = build_regressor,
                           batch_size = 32, epochs = 110)

parameters = {'optimizer': ['nadam','rmsprop', 'adam']}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
"""

 
