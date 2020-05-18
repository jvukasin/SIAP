import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


def xbg_algoritam(X_train, y_train, X_test, y_test):
    # fit model no training data
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=4, subsample=0.8, colsample_bytree=0.7,
                              gamma=1)

    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Validation accuracy XGBoost: %.2f%%" % (accuracy * 100.0))

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score XGBoost: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_pred)
    print("R2 score XGBoost: %.2f" % r2)

    mean = mean_squared_error(y_test, y_pred)
    print("Mean squared error XGBoost: %.2f" % mean)

    rmean = np.sqrt(mean)
    print("Root mean squared error XGBoost: %.2f" % rmean)
