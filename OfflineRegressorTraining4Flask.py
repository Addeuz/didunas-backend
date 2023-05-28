# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:43:25 2023

@author: fanha
"""

import pandas as pd
import numpy as np

def Load_Training_data(responseTime_sheet_file='DIDUNAS_main-study_responseTimeVsPvalue.xlsx',
                       accuracy_sheet_file='DIDUNAS_main-study_errorRateVsPvalue.xlsx', num_tasks=10):

    responseTime_sheet = pd.read_excel(responseTime_sheet_file, engine='openpyxl')
    accuracy_sheet = pd.read_excel(accuracy_sheet_file, engine='openpyxl')

    responseTime_mat = responseTime_sheet.loc[:, :]
    accuracy_mat = accuracy_sheet.loc[:, :]

    num_users = responseTime_mat.shape[0]

    X = np.zeros( (num_users, num_tasks) )

    for u in range(num_users):
        for t in range(num_tasks):
            X[u, t] = accuracy_mat.loc[u][t+1]
    
    y = responseTime_mat['p-value']
    yr= np.copy(y)

    threshold = 0.192
    yc = np.copy(y)
    yc[yc>=threshold] = 1
    yc[yc<threshold] = 0
    
    return X, yc, yr

from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors,tree
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from imblearn.over_sampling import SMOTE
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier 


def Hybrid_model_training(X_train, yr_train, yc_train, SMOTE_sam=0.4, num_features=6):
    
    trans_reg = TransformedTargetRegressor(
    regressor=tree.DecisionTreeRegressor(max_depth=7),# neighbors.KNeighborsRegressor() is outperformed
    transformer=QuantileTransformer(n_quantiles=50, output_distribution="normal"),)
    
    # define the model
    reg_model = AdaBoostRegressor(
    trans_reg,
    n_estimators = 500,
    loss='exponential',)
    
    adaR_model = reg_model.fit(X_train, yr_train)

    log_classifier = LogisticRegression(penalty='l2', max_iter=2000)
    cla_model = AdaBoostClassifier(
       log_classifier,
       n_estimators = 500,
       )

    over = SMOTE(sampling_strategy=0.35)
   
    ABC_pipe = Pipeline([ ('over', over), ('pca', PCA(6)), ('AdaBoostClassifier', cla_model) ])
    
    # execute search
    adaC_model = ABC_pipe.fit(X_train, yc_train)
   
    return adaR_model, adaC_model

import pickle
import numpy as np

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

        return yhat

    
    
model_version = '202305'
if __name__ == '__main__':
    X, yc, yr = Load_Training_data(responseTime_sheet_file='DIDUNAS_main-study_responseTimeVsPvalue.xlsx',
                       accuracy_sheet_file='DIDUNAS_main-study_errorRateVsPvalue.xlsx', 
                       num_tasks=10)
    
    
    adaR_model, adaC_model = Hybrid_model_training(X, yr, yc)
    
    # Create an instance of the HybridEnsembleModel class
    hybrid_model = HybridEnsembleModel(adaC_model, adaR_model)
    
    pickle.dump(hybrid_model, open('DIDUNAS_regression_model'+str(model_version)+'.pkl','wb'))
   
    