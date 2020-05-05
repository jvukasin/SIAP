from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def gbt_algorythm(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=20, random_state=42, learning_rate=0.1, max_depth=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score Gradient boosted tree: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_pred)
    print("R^2 score Gradient boosted tree: %.2f" % r2)

    mean = mean_squared_error(y_test, y_pred)
    print("Mean squared error Gradient boosted tree: %.2f" % mean)

    rmse = np.sqrt(mean)
    print("Root mean square error Gradient boosted tree: %.2f" % rmse)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report")
    print(classification_report(y_test, y_pred))

