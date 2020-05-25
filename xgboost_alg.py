import xgboost as xgb
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def xbg_algoritam(X_train, y_train, X_test, y_test, features):

    # use_gridSearch(X_train, y_train)

    model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=4)

    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]


    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score XGBoost: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_pred)
    print("R2 score XGBoost: %.2f" % r2)

    mean = mean_squared_error(y_test, y_pred)
    rmean = np.sqrt(mean)
    print("Root mean squared error XGBoost: %.2f" % rmean)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    # plot feature importance from dataset
    feature_importance(model, features)


def feature_importance(forest, features):
    importances = forest.feature_importances_
    indices = np.argsort(importances)

    plt.figure(1)
    plt.title('XGBoost feature importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()


def use_gridSearch(X_train, y_train):
    num_estimators = [50, 100]
    learn_rates = [0.5, 0.1, 0.2, 0.3]
    max_depths = [4, 6, 8, 10]

    param_grid = {'n_estimators': num_estimators,
                  'learning_rate': learn_rates,
                  'max_depth': max_depths}

    grid_search = GridSearchCV(xgb.XGBClassifier(),
                               param_grid, cv=5, return_train_score=True, verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
