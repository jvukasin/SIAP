import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
import numpy as np


def xbg_algoritam(X_train, y_train, X_test, y_test):
    # fit model no training data
    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Validation accuracy XGBoost: %.2f%%" % (accuracy * 100.0))

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score XGBoost: %.2f%%" % (variance_score * 100.0))

