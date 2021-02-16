# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:24:47 2021

@author: mshahzamal
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

#naming our app as app
app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features= [float(x) for x in request.form.values()]
    final_features= [np.array(int_features)]
    prediction= model.predict(final_features)
    output= round(prediction[0], 2)
    return render_template("index.html", prediction_text= "flower is {}".format(output))

#running the flask app

if __name__ == '__main__':
    app.run(host='0.0.0.0')