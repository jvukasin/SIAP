import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


def svr_algorithm(X_train, y_train, X_test, y_test):

    model = SVR(kernel='rbf', gamma='scale', C=1000) # epslion = 0.1 ne menja nista

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score SVR: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_pred)
    print("R^2 score SVR: %.2f" % r2)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean square error SVR: %.2f" % mse)

    rmse = np.sqrt(mse)
    print("Root mean square error SVR: %.2f" % rmse)


