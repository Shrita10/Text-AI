# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:30:21 2021

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
from flask import Flask, jsonify, json
import mysql.connector
from mysql.connector import MySQLConnection

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def create_conn():
    conn = MySQLConnection(host = "127.0.0.1" , user = "root" ,password = "1243" , database = "textai", auth_plugin='mysql_native_password')
    return conn



@app.route('/')

def home():
    return render_template('updated.html')

@app.route('/predict2',methods = ['POST'])

def predict2():
    input_text = request.form['ttext']
    option = request.form['select10']
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO textai.texttable('input_text','option') VALUES (%s,%s)",(input_text,option))        
    conn.commit()
    cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
    tokens = word_tokenize(cleaned)
    train_len = 1
    text_sequences = []  
    for i in range(train_len,len(tokens)):
        text_sequences.append(tokens[i])
    sequences = {}
    count = 1
    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    input_text = input_text.strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
    list_of_words =[]
    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
        pred_word = tokenizer.index_word[i]
        list_of_words.append(pred_word)
        print("Next word suggestion:",pred_word) 
    first_word = list_of_words[0]
    second_word = list_of_words[1]
    third_word = list_of_words[2]
    return render_template('html1.html',prediction_text1 = first_word , prediction_text2 =  second_word, prediction_text3 = third_word)
   

    





if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    