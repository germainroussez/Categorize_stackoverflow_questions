# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:00:50 2021

@author: germa
"""

from flask import Flask, render_template, request
import pickle
import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

multilabel_binarizer = pickle.load(open('Saves/multilabel_binarizer.pkl', 'rb'))
lp_classifier = pickle.load(open('Saves/lp_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('Saves/custom_vectorizer.pkl', 'rb'))
en_nlp = spacy.load('en_core_web_trf')

def remove_convert_non_letters(text):
    text = re.sub("c\+\+", "cplusplus", text)
    text = re.sub("c#", "csharp", text)
    text = re.sub("\.net", "dotnet", text)
    text = re.sub("d3\.js", "d3js", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.strip(u'\n')
    return text

def convert_to_lowercase_letters(text):
    return text.lower()

def custom_tokenizer(text):
    text_spacy = en_nlp(text)
    tokens_lemma = [token.lemma_ for token in text_spacy]
    return ' '.join(tokens_lemma)

def remove_stopwords(text):
    text_tokens = text.split()
    tokens_without_sw = [word for word in text_tokens if not word in ENGLISH_STOP_WORDS]
    return " ".join(tokens_without_sw)

def custom_vectorizer(text):
    BOW = vectorizer.transform(text)
    return BOW

def pred(text):
    non_letters_removed = remove_convert_non_letters(text)
    only_lowercase_letters = convert_to_lowercase_letters(non_letters_removed)
    text_tokenized = custom_tokenizer(only_lowercase_letters)
    text_without_stopwords = remove_stopwords(text_tokenized)
    BOW = custom_vectorizer([text_without_stopwords])
    lp_prediction = lp_classifier.predict(BOW)
    lp_prediction_labels = multilabel_binarizer.inverse_transform(lp_prediction)
    print(lp_prediction_labels)
    return lp_prediction_labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = ' '.join([x for x in request.form.values()])
    output = pred(features)
    return render_template('index.html', prediction_text='The predicted tag(s) for the given subject could be : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)