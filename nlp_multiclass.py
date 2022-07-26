# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:13:14 2022

@author: eshan
"""
import re
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from nlp_multiclass_module import ModelDevelopment
from nlp_multiclass_module import ModelEvaluation

#%% Constants
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(),'model','model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'model','tokenizer.json')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe.pkl')

#%% Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

#%% EDA

df.info()
df.describe().T
df.duplicated().sum() #99 duplicates

#temporary dataframe to visualise duplicated text
temp = df[df.duplicated(keep=False) == True]
temp = temp.sort_values(by=['text'])

# Symbols (eg .,-) present in text data

#%% Data Cleaning

# #Remove duplicates
df = df.drop_duplicates()
df.duplicated().sum() # no duplicates
df = df.reset_index(drop=True)

# Remove symbols & split text
category = df['category']
text = df['text']

for i, content in enumerate(text):
    text[i] = re.sub('[^a-zA-Z]',' ',content).split()
 
#%% Data Preprocessing

# X Features
vocab_size = 10000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10])) #slice to inspect first 10 values

# Save tokenizer
tokenizer_json = tokenizer.to_json()

with open(TOKENIZER_PATH, "w") as file:
    json.dump(tokenizer_json,file)

# Change text to integers
text_int = tokenizer.texts_to_sequences(text)

length_text=[]

for i in range(len(text_int)):
    length_text.append(len(text_int[i]))

max_len = np.median(length_text) # 333

# Add padding
padded_text = pad_sequences(text_int, maxlen=int(max_len), 
                              padding='post', 
                              truncating='post')

# Y target
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))

# Save encoder
with open(OHE_PATH, "wb") as file:
    pickle.dump(ohe,file)

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(padded_text, category,
                                                    test_size=0.3, 
                                                    random_state=123)

#%% Model Development

md = ModelDevelopment()

model = md.nlp_model(X_train, y_train, vocab_size)

plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

#%%

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

earlystopping = EarlyStopping(monitor='val_loss', patience=3)

#save model with lowest val_loss to reduce overfitting
mcp = ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, 
                      monitor='val_loss', mode='min', verbose=1)

hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, 
                 callbacks=[tensorboard_callback,earlystopping, mcp])

#%% Model Evaluation

# load previously saved model
model = load_model(MODEL_PATH)

# plot loss & accuracy training graphs

me = ModelEvaluation()
me.plot_hist_graph(hist)

print(model.evaluate(X_test, y_test))

# loss: 0.29, acc: 0.91

#%%

# classification report
pred_y = model.predict(X_test)

pred_y = np.argmax(pred_y,axis=1)

true_y = np.argmax(y_test,axis=1)

print(classification_report(true_y,pred_y))

# confusion matrix
conmx = confusion_matrix(true_y,pred_y)
disp = ConfusionMatrixDisplay(confusion_matrix=conmx)
disp.plot(cmap=plt.cm.Blues)
plt.show()
