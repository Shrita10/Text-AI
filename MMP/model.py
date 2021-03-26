# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 05:01:02 2021

@author: Shrita
"""


from keras.preprocessing.text import Tokenizer
import nltk
import pickle
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
from doc3 import training_doc3

cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
tokens = word_tokenize(cleaned)
train_len = 3+1
text_sequences = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)
sequences = {}
count = 1

for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
        
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 

#Collecting some information   
vocabulary_size = len(tokenizer.word_counts)+1

n_sequences = np.empty([len(sequences),train_len], dtype='int32')

for i in range(len(sequences)):
    n_sequences[i] = sequences[i]
    
    
train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]
train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
seq_len = train_inputs.shape[1]

print(seq_len)
train_inputs.shape
#print(train_targets[0])

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
#model = load_model("mymodel.h5")

model = Sequential()
model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50,activation='relu'))
model.add(Dense(vocabulary_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_inputs,train_targets,epochs=10,verbose=1)
model.save("mymodel.h5")

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(sequences, open('tokenizer.pkl', 'wb'))

print("hello")

from keras.preprocessing.sequence import pad_sequences
#input_text = input().strip().lower()
input_text = "ass"
print(input_text)
encoded_text = tokenizer.texts_to_sequences([input_text])[0]
pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
list_of_words =[]
for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    print(tokenizer.index_word)
    pred_word = tokenizer.index_word[i]
    list_of_words.append(pred_word)
    print("Next word suggestion:",pred_word)
    
first_word = list_of_words[0]
second_word = list_of_words[1]
third_word = list_of_words[2]