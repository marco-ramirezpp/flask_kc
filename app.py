from flask import Flask,jsonify,request
import pickle
import os
from flasgger import Swagger
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
Swagger(app)


@app.route('/predict', methods=['GET'])
def get_predictions():
    """
    Modelo KC
    ---


    parameters:
       - name: feature_1
         in: query
         type: number
         required: true
       - name: feature_2
         in: query
         type: number
         required: true
       - name: feature_3
         in: query
         type: number
         required: true

       - name: feature_4
         in: query
         type: number
         required: true

       - name: feature_5
         in: query
         type: number
         required: true

       - name: feature_6
         in: query
         type: number
         required: true

       - name: feature_7
         in: query
         type: number
         required: true

       - name: feature_8
         in: query
         type: number
         required: true

       - name: feature_9
         in: query
         type: number
         required: true

       - name: feature_10
         in: query
         type: number
         required: true

       - name: feature_11
         in: query
         type: number
         required: true

       - name: feature_12
         in: query
         type: number
         required: true


    responses:


        200:
            description : predicted Class
    """

    ## Getting Features from Swagger UI
    feature_1 = int(request.args.get("feature_1"))
    feature_2 = int(request.args.get("feature_2"))
    feature_3 = int(request.args.get("feature_3"))
    feature_4 = int(request.args.get("feature_4"))
    feature_5 = int(request.args.get("feature_5"))
    feature_6 = int(request.args.get("feature_6"))
    feature_7 = int(request.args.get("feature_7"))
    feature_8 = int(request.args.get("feature_8"))
    feature_9 = int(request.args.get("feature_9"))
    feature_10 = int(request.args.get("feature_10"))
    feature_11 = int(request.args.get("feature_11"))
    feature_12 = int(request.args.get("feature_12"))

    #    feature_1 = 1
    #    feature_2 = 2
    #    feature_3 = 3
    #    feature_4 = 4
    #    feature_5 = 5
    #    feature_6 = 6
    #    feature_7 = 7
    #    feature_8 = 8
    #    feature_9 = 9
    #    feature_10 = 10
    #    feature_11 = 11
    #    feature_12 = 12

    headers = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_lot',
       'waterfront', 'view', 'condition', 'grade', 'lat', 'long']
    test_set = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6,
                          feature_7, feature_8, feature_9, feature_10, feature_11, feature_12]],
                                   columns=headers,
                                   dtype=float,
                                   index=['input'])


    ## Loading Model
    infile = open('kc_model-091321.pkl', 'rb')
    model = joblib.load('kc_model-091321.pkl')
    infile.close()

    ## Generating Prediction
    preds = model.predict(test_set)

    return jsonify({"class_name": str(preds)})

if __name__ == '__main__':
    app.run(debug=True)
