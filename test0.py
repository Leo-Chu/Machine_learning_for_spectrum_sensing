# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:10:38 2019

@author: LeoChu
"""

import numpy as np
from SpectrumkerasS import load_train_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense



def makedata():
    dataX, labelX =  load_train_data()
    nb_class = 2
    label = np_utils.to_categorical(labelX, nb_class)
    
    
    x1,y1 = dataX[:7000,np.newaxis,:], label[:7000,:]
    x2,y2 = dataX[10000:17000,np.newaxis,:], label[10000:17000,:]
    
    x3,y3 = dataX[7000:10000,np.newaxis,:], label[7000:10000,:]
    x4,y4 = dataX[17000:,np.newaxis,:], label[17000:,:]
    
    x_train, y_train = np.concatenate((x1,x2)),np.concatenate((y1,y2))
    x_test, y_test = np.concatenate((x3,x4)),np.concatenate((y3,y4))
    
    
    
    return x_train, y_train, x_test, y_test
    
    
def buildlstm():

    data_dim = 100
    timesteps = 1
    num_classes = 2

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,   input_shape=(timesteps, data_dim)))   
    model.add(LSTM(32, return_sequences=True))  
    model.add(LSTM(32))  
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
#    print model.summary()
    return  model

def runTrain(model, x_train, x_test, y_train, y_test):
    print('Start Training!')

    model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 10) # ,shuffle=True
    print('Training is over and Start to Test!')
    score = model.evaluate(x_test, y_test, batch_size = 10)
    print('Test accuracy:', score[1])
   

def test():

    x_train, y_train, x_test, y_test = makedata()
    model = buildlstm()
    runTrain(model, x_train, x_test, y_train, y_test )
    
    
if __name__ == "__main__":  
     test()
     
     
     
     
     
     