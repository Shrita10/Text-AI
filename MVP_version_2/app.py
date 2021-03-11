# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 03:49:28 2021

@author: Shrita
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from doc3 import training_doc3

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


@app.route('/')

def home():
    return render_template('html1.html')

@app.route('/predict2',methods = ['POST'])
def predict2():
    ''' For results on HTML GUI'''
    
    input_text = request.form['ttext']
    cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
    tokens = word_tokenize(cleaned)
    
    print(tokens)
    train_len = 1
    text_sequences = []
    
    for i in range(train_len,len(tokens)):
        #seq = tokens[i-train_len:i]
        #text_sequences.append(seq)
        text_sequences.append(tokens[i])
    sequences = {}
    count = 1
    
    print(text_sequences)
    
    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
         
    tokenizer = Tokenizer()
    print(text_sequences)
    tokenizer.fit_on_texts(text_sequences)
    #sequences = tokenizer.texts_to_sequences(text_sequences) 
    #print("This is pad sequences", sequences)
        
    # encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    
    
    input_text = input_text.strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
    list_of_words =[]
    
    print("This is pad encoded", pad_encoded)
    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
        print(tokenizer.index_word)
        pred_word = tokenizer.index_word[i]
        list_of_words.append(pred_word)
        print("Next word suggestion:",pred_word)
        
    #for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    
    '''var1 = model.predict(pad_encoded[0][1])
    print("this is var", var1)
    var2 = var1[0]
    var3 = var2.argsort()
    var4 = var3[-3:][::-1]
    
    for i in var4:
        #pred_word = tokenizer.index_word[i]
        print(tokenizer.index_word)
        pred_word = tokenizer.index_word[0]
        list_of_words.append(pred_word)
        print("Next word suggestion:",pred_word)'''
        
        
    
        
    first_word = list_of_words[0]
    second_word = list_of_words[1]
    third_word = list_of_words[2]


    return render_template('html1.html',prediction_text1 = 'The predicted word is {}'.format(first_word), prediction_text2 = 'The second predicted word is {}'.format(second_word), prediction_text3 = 'The third predicted word is {}'.format(third_word))
    

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    