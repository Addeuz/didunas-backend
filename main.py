# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request
import pickle
import numpy as np
from flask_cors import CORS
import logging
import __main__

class HybridEnsembleModel:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def hybrid_prediction(self, X_test):
        if X_test.shape[1] < 2:
            y_flag = self.classifier.predict(X_test.reshape([1, 10]))
            yhat = self.regressor.predict(X_test.reshape([1, 10]))

            if y_flag > 0 and yhat < 0.25:
                yhat = np.sqrt(yhat)
            if y_flag < 1 and yhat > 0.25:
                yhat = yhat * yhat

        else:
            y_flag = self.classifier.predict(X_test)
            yhat = self.regressor.predict(X_test)

            for count, value in enumerate(y_flag):
                if value > 0 and yhat[count] < 0.25:
                    yhat[count] = np.sqrt(yhat)
                if value < 1 and yhat[count] >= 0.25:
                    yhat[count] = yhat[count] * yhat[count]

        return y_flag, yhat
__main__.HybridEnsembleModel = HybridEnsembleModel

logging.getLogger('flask_cors').level = logging.DEBUG

app=Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "https://madita.vercel.app"}})

model = pickle.load(open('DIDUNAS_regression_model202305.pkl','rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    predictions = {}

    for user_id, value in data.items():
        features_list = [
            value['quantities'],
            value['quantityComparison'],
            value['numberComparison'],
            value['colorPattern'],
            value['numberPattern'],
            value['hiddenNumber'],
            value['numberLine'],
            value['completionToTen'],
            value['plus'],
            value['minus'],
        ]
        features = np.array(features_list).reshape([1,10])
        predict_RMD, predict_pValue = model.hybrid_prediction(features) #note that here both Y/N predciton and a P-value predciotion are made.

        if predict_RMD>0:
            RMD='T'
        else:
            RMD='F'

        predictions[user_id] = RMD

    return predictions
