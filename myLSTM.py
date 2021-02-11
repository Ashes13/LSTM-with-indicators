import numpy as np
import pandas as pd

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class MyLSTM:
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range = (0, 1))

    def ProcessData(self, data, n = 30):
        # Split the data
        training_data = data[:1260]
        test_data = data[1260:]


        train_array = np.array(training_data).reshape((len(training_data), 3))
        
        #scale features 
        spy_training_scaled = self.scaler.fit_transform(train_array)
        
        
        # Build feauture and label ets (using number of steps 60, batch size 1200, and hidden size 1)
        features_set = []
        labels = []
        for i in range(30, 1260):
            #gets lagged 60 timesteps of features
            features_set.append(spy_training_scaled[i-30:i, 0:3])
            #gets lagged close price (labels)
            labels.append(spy_training_scaled[i,2])
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 3))
        
        
        return features_set, labels, training_data, test_data

        
    
    def CreateModel(self, features_set, labels):
        
        # Build a Sequential keras model
        self.model = Sequential()
        # Add our first LSTM layer - 50 nodes
        self.model.add(LSTM(units = 50, return_sequences=True, input_shape=(features_set.shape[1], 3)))
        # Add Dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))
        # Add additional layers
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        # Compile the model
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])  
        
        # Fit the model to our data, running 100 training epochs

    def FitModel(self, combined_features, combined_labels):
        self.model.fit(combined_features, combined_labels, epochs = 15, batch_size = 32)
    
        
    def PredictFromModel(self, test_data,test_labels):

        test_inputs = test_data[-80:]
        train_array = np.array(test_inputs).reshape((len(test_inputs), 3))
        test_inputs = self.scaler.transform(train_array)
        
        #test close for scalar inverse of predictions
        test_close  = test_labels[-80:].values
        test_close_array = np.array(test_close).reshape((len(test_close), 1))
        testscaler = MinMaxScaler(feature_range = (0, 1))
        test_scaled_array = testscaler.fit_transform(test_close_array)
        
    
        test_features = []
        for i in range(30, 80):
            test_features.append(test_inputs[i-30:i, 0:3])
        test_features = np.array(test_features)
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 3))
    
        predictions = self.model.predict(test_features)
        predictions = testscaler.inverse_transform(predictions)
        return predictions
        
