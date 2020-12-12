# By Howard
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:06:03 2017
@author: Howard
"""

import pandas as pd
import numpy as np
# import keras's Sequential model
from keras.models import Sequential
#from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras.layers import Dense,Dropout
from sklearn import preprocessing
from keras import backend as K
from keras.callbacks import ModelCheckpoint

np.random.seed(10)

# read data
data_train = pd.read_csv('./regression/train-v3.csv')
x_train = data_train.drop(['price','id'],axis=1).values
y_train = data_train['price'].values
#y_train = y_train.reshape((-1, 1))

data_valid = pd.read_csv('./regression/valid-v3.csv')
x_valid = data_valid.drop(['price','id'],axis=1).values
y_valid = data_valid['price'].values
#y_valid = y_valid.reshape((-1, 1))

data_test = pd.read_csv('./regression/test-v3.csv')
x_test = data_test.drop(['id'],axis=1).values
#test_id = data_valid['id'].values
#print(x_test[:1])

# sqft_living = data_test['sqft_living'].values

# data normalize use sklearn Preprocess model
"""
min_max_scaler = preprocessing.MinMaxScaler()   
x_train_minmax = min_max_scaler.fit_transform(x_train)    
x_valid_minmax = min_max_scaler.fit_transform(x_valid) 
x_test_minmax = min_max_scaler.fit_transform(x_test)  
"""
 
X_train = preprocessing.scale(x_train)
X_valid = preprocessing.scale(x_valid)
X_test = preprocessing.scale(x_test)

print('x_train=',x_train.shape)

# create keras's Sequential model

# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model = Sequential()
model.add(Dense(units=16,input_dim=x_train.shape[1], kernel_initializer='random_normal',activation='relu'))
# model1
# model.add(Dense(units=10, kernel_initializer='random_normal',activation='relu'))
# model.add(Dense(units=200, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=200, kernel_initializer='random_normal',activation='relu'))
# model.add(Dense(units=300, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=300, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# # model.add(Dense(units=256, kernel_initializer='random_normal',activation='relu'))
# # model.add(Dense(units=256, kernel_initializer='random_normal',activation='relu'))
# model.add(Dense(units=300, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=300, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=300, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=200, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=200, kernel_initializer='random_normal',activation='relu'))
# model.add((Dropout(0.3)))
# model.add(Dense(units=50, kernel_initializer='random_normal',activation='relu'))
#==============================================================================
# model2
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=64, kernel_initializer='random_normal',activation='relu'))
model.add(Dense(units=16, kernel_initializer='random_normal',activation='relu'))
#==============================================================================
model.add(Dense(units=x_train.shape[1], kernel_initializer='random_normal',activation='relu'))
# output layer
model.add(Dense(units=1, kernel_initializer='random_normal'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

print(model.summary())
# model.compile(loss='MAE',optimizer=adam)
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)

model.compile(loss='MAE',optimizer='Adam')

#儲存權重
checkpoint = ModelCheckpoint('weight/ep{epoch:03d}-loss{loss:03f}-val_loss{val_loss:.3f}.h5', monitor='val_loss',verbose=1, save_best_only=True)

train_history = model.fit(x=X_train,y=y_train,validation_data=(X_valid,y_valid),epochs=2000,batch_size=5000,callbacks=[checkpoint])

Y_predict = model.predict(X_test)

np.savetxt('test.csv',Y_predict,delimiter=',')
#output = np.column_stack((test_id, Y_predict))
#np.savetxt('test200.csv', output, delimiter=',', header='id,price', comments='')

# Using pyplot
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()
    
show_train_history(train_history, 'loss', 'val_loss')