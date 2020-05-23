import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


def xbg_algoritam(X_train, y_train, X_test, y_test, features):

    # num_estimators = [20, 100]
    # learn_rates = [0.05, 0.5]
    # max_depths = [1, 5]
    # min_samples_leaf = [2, 10]
    # min_samples_split = [2, 10]
    #
    # param_grid = {'n_estimators': num_estimators,
    #               'learning_rate': learn_rates,
    #               'max_depth': max_depths,
    #               'min_samples_leaf': min_samples_leaf,
    #               'min_samples_split': min_samples_split}
    #
    # grid_search = GridSearchCV(xgb.XGBClassifier(),
    #                            param_grid, cv=5, return_train_score=True)
    #
    # grid_search.fit(X_train, y_train)
    #
    # print(grid_search.best_params_)

    # training model with best params

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=4, subsample=0.8, gamma=1)

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

    # plot feature importance from dataset
    feature_importance(model, features)


def feature_importance(forest, features):
    importances = forest.feature_importances_
    indices = np.argsort(importances)

    plt.figure(1)
    plt.title('Random Forest feature importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()
