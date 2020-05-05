from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report, confusion_matrix


def gbt_algorythm(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=20, random_state=42, learning_rate=0.1, max_depth=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    variance_score = explained_variance_score(y_test, y_pred)
    print("Explained variance score XGBoost: %.2f%%" % (variance_score * 100.0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report")
    print(classification_report(y_test, y_pred))

