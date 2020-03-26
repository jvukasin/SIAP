from sklearn.ensemble import RandomForestClassifier #RF klasifikator
from sklearn.metrics import accuracy_score


def rf_algoritam(X_train, y_train, X_test, y_test):

    data_master_RF = RandomForestClassifier(n_estimators=50, random_state=42)
    data_master_RF = data_master_RF.fit(X_train, y_train)
    y_train_pred = data_master_RF.predict(X_train)
    y_test_pred = data_master_RF.predict(X_test)
    print("Train accuracy RF: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy RF: ", accuracy_score(y_test, y_test_pred))

    print(y_test)
    print(y_test_pred)

