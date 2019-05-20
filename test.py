# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:47:31 2019

@author: dell
"""

import numpy as np
#import random
from keras.utils import np_utils
#from keras.models import Sequential

from SpectrumkerasS import load_train_data
#import time
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy.io as sio
#from sklearn.svm import SVR
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier    #RandomForestClassifier
from sklearn.svm import SVC
#from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import tree
from sklearn import metrics
#from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import LinearSVC


dataX, labelX =  load_train_data()
nb_class = 2
label = np_utils.to_categorical(labelX, nb_class)


x1,y1 = dataX[:7000,:], label[:7000,:]
x2,y2 = dataX[10000:17000,:], label[10000:17000,:]

x3,y3 = dataX[7000:10000,:], label[7000:10000,:]
x4,y4 = dataX[17000:,:], label[17000:,:]

x_train, y_train = np.concatenate((x1,x2)),np.concatenate((y1,y2))
x_test, y_test = np.concatenate((x3,x4)),np.concatenate((y3,y4))


#######GradientBoostingClassifier

gbm = GradientBoostingClassifier(learning_rate=0.05, 
                                  n_estimators=120,
                                  max_depth=7, 
                                  min_samples_leaf =60, 
                                  min_samples_split =1200, 
                                  max_features=9, 
                                  subsample=0.7, random_state=10)
gbm.fit(x_train, y_train[:,1])
y_pred = gbm.predict(x_test)
accuracy_gbm = metrics.accuracy_score(y_test[:,1], y_pred)
print("Accuracy of GBM is:", accuracy_gbm)


### RF classifier
max_depth = 1000
#
regr_rf = RandomForestClassifier(n_estimators = 144, \
                                max_depth=max_depth, random_state=0)
regr_rf.fit(x_train, y_train)

y_pred = regr_rf.predict(x_test)

#target_names = ['Signal_Detect', 'Noise_Detect']
#classification_report(y_test, res1, target_names = target_names)

accuracy_rf = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of RF classifier is:", accuracy_rf)

####  SVC


poly_kernel_svm_clf = Pipeline([ ( "scaler", StandardScaler()),
                                 ("svm_clf", SVC(kernel="poly", 
                                                 degree=3, coef0=1, 
                                                 C=0.5))
                                ])
poly_kernel_svm_clf.fit(x_train, y_train[:,1])


y_pred = poly_kernel_svm_clf.predict(x_test)

accuracy_svc = metrics.accuracy_score(y_test[:,1], y_pred)

print("Accuracy of SVC is:", accuracy_svc)

#clt x1,x2,x3,x4,y1,y2,y3,y4

#
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
