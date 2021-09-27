# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:00:50 2021

@author: germa
"""

from flask import Flask, render_template,redirect, request

app = Flask(__name__)

def pred(features):
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features = [x for x in features[0].split()] + [y for y in features[1].split()]
    output = pred(features)
    return render_template('index.html', prediction_text='The predicted tag(s) for the given subject could be : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)