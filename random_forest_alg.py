from sklearn.ensemble import RandomForestClassifier  # RF klasifikator
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, f1_score, precision_score, \
    recall_score
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def rf_algoritam(X_train, y_train, X_test, y_test, features):
    # use_gridSearch(X_train, y_train)

    data_master_RF = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=100, min_samples_split=4)
    model = data_master_RF.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)


    '''Cross-validation'''
    cv = cross_val_score(model, X_train, y_train, cv=5)
    print('cross validation')
    print(cv)
    print("Mean of cv: ", cv.mean())
    print("Std of cv: ", cv.std())
    cv_pred = cross_val_predict(model, X_train, y_train, cv=5)
    accCV = r2_score(y_train, cv_pred)
    print("R^2 score CV: %.2f" % accCV)

    # print("Train accuracy RF: ", accuracy_score(y_train, y_train_pred))
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Validation accuracy RandomForest: %.2f%%" % (accuracy * 100.0))

    variance_score = explained_variance_score(y_test, y_test_pred)
    print("Explained variance score RandomForest: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_test_pred)
    print("R^2 score RandomForest: %.2f" % r2)

    mean = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean)
    print("Root mean square error RandonForest: %.2f" % rmse)

    # print('razlika: ', set(y_test) - set(y_test_pred))
    f1 = f1_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred))
    print('f1 score RandomForest: ', f1)

    precision = precision_score(y_test, y_test_pred, average='micro')
    print('precision score RandomForest: ', precision)

    recall = recall_score(y_test, y_test_pred, average='micro')
    print('recall score RandomForest: ', recall)

    # slika_stabla_RF(data_master_RF, feature_list)

    # plot feature importance from dataset
    feature_importance(data_master_RF, features)


def feature_importance(forest, features):
    importances = forest.feature_importances_
    indices = np.argsort(importances)

    plt.figure(1)
    plt.title('Random Forest feature importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()

def use_gridSearch(X_train, y_train):
    n_estimators = [20, 50, 100, 200]
    min_samples_split = [2, 3, 4, 5, 6]
    max_depth = [2, 5, 10, 20, 50, 100, 200]
    param_grid = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, return_train_score='true', verbose=2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
