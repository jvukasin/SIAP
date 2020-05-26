import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.metrics import classification_report, confusion_matrix


def svr_algorithm(X_train, y_train, X_test, y_test):

    # use_gridSearch(X_train, y_train)

    model = SVR(kernel='rbf', C=1000) # epslion = 0.1 ne menja nista

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score SVR: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_pred)
    print("R^2 score SVR: %.2f" % r2)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Root mean square error SVR: %.2f" % rmse)


def use_gridSearch(X_train, y_train):
    C = [1, 10, 100, 200, 500, 1000]
    kernel = ['rbf', 'poly', 'sigmoid']
    gamma = ['scale']

    param_grid = {'C': C,
                  'kernel': kernel,
                  'gamma': gamma}

    grid_search = GridSearchCV(SVR(),
                               param_grid, cv=5, return_train_score=True, verbose=2, n_jobs=2)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)



