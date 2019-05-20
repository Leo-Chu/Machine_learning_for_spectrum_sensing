from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
import os
#import cv2
#import h5py 
import scipy.io as sio
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.layers import LSTM

def load_data(path):
#    print(path)
    dd = os.listdir(path)
    mats = ''.join(dd)

    row, col = 100, 10000
    dataMat = np.empty((row, col), dtype="uint8")
    each_data = sio.loadmat(path + mats)
    
    dataMat = each_data["data"]  
    return dataMat.T 
     
def load_train_data():
#   folders = ["./HC/","./FES/","./CHR/"]       

   folders = ["./Noise/","./Signal/"] 
   folders_num = len(folders)
#   print('folders_num', folders_num)
   dataSet = np.empty((), dtype="uint8")
   label = np.empty((), dtype="uint8")

   for i in range(folders_num):
       data = load_data(folders[i])
       each_label = np.empty((data.shape[0],1), dtype="uint8")
       each_label.fill(i)
       if i == 0:
           dataSet, label = data, each_label
       else:
           print(dataSet.shape)
           dataSet = np.concatenate((dataSet, data))
           label = np.concatenate((label, each_label))
           
   return dataSet, label

   
if __name__ == "__main__":  
     data, label = load_train_data()
    

#import random
#random.shuffle(data, label) 
#    
#x_train, y_train, x_test, y_test = load_data() 
#
#model = Sequential()
#LSTM_layer_sizes = 2
#
#epoch_no = 100
#batch_size = 500
#
#lr = 0.001
#lr_decay = 0.1
#
#for size in LSTM_layer_sizes[:-1]:
#	model.add(LSTM(units=size, return_sequences=True,
#				   recurrent_dropout=recurrent_dropout_factor,
#				   dropout=LSTM_dropout_factor))
#	model.add(Dropout(layer_dropout_factor))
#model.add(LSTM(units=LSTM_layer_sizes[-1], recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
#model.add(Dropout(layer_dropout_factor))
#model.add(Dense(y_train.shape[1], activation='sigmoid'))
#optimizer = Adam(lr=lr, decay=lr_decay)
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy', c_score])
#print(model.summary())
#model.fit(X_train, y_train, validation_data=(x_test, y_test), epochs=epoch_no, batch_size=batch_size,
#		  callbacks=[ModelCheckpoint("weights.hdf5", monitor='val_loss',
#									 save_best_only=True, mode='auto', period=1),
#					 LogPerformance()])
#
#scores = model.evaluate(x_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
#
#
#
