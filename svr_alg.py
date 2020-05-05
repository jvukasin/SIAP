import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error


def svr_algorithm(X_train, y_train, X_test, y_test):
    model = SVR(kernel='rbf')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Mean square error: %.2f%%" % mse)
    print("Root mean square error: %.2f%%" % rmse)

    r2 = r2_score(y_test, y_pred)
    print("R2 score: %.2f%%" % (r2 * 100.0))

