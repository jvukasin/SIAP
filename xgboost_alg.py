import xgboost as xgb
from sklearn.metrics import accuracy_score


def xbg_algoraim(X_train, y_train, X_test, y_test):
    # fit model no training data
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

