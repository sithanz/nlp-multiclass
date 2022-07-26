# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:25:16 2022

@author: eshan
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout, Embedding

#%%
class ModelDevelopment:
    def nlp_model(self, X_train, y_train, vocab_size, 
                  out_dim=64, 
                  dropout_rate=0.5):
        
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Embedding(vocab_size, out_dim))
        model.add(Bidirectional(LSTM(out_dim, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(out_dim)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(np.shape(y_train)[1],activation='softmax'))
        model.summary()
        return model
    
    
class ModelEvaluation:
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history[list(hist.history.keys())[0]])
        plt.plot(hist.history[list(hist.history.keys())[2]])
        plt.xlabel('epoch')
        plt.legend([list(hist.history.keys())[0],list(hist.history.keys())[2]])
        plt.show()

        plt.figure()
        plt.plot(hist.history[list(hist.history.keys())[1]])
        plt.plot(hist.history[list(hist.history.keys())[3]])
        plt.xlabel('epoch')
        plt.legend([list(hist.history.keys())[1],list(hist.history.keys())[3]])
        plt.show()
        pass