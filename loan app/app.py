#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/loan', methods=['POST', 'GET'])
def loan():
    return render_template('loan.html')

@app.route('/loan.html', methods=['POST', 'GET'])
def rloan():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        pred = "APPROVED !!!"
    elif prediction == 0:
        pred = "REJECTED !!!"
    output = pred
    return render_template('loan.html', prediction_text=output)

@app.route('/back', methods=['POST', 'GET'])
def back():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

