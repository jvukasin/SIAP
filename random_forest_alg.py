from sklearn.ensemble import RandomForestClassifier #RF klasifikator
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.tree import export_graphviz
import pydot
import numpy as np
import matplotlib.pyplot as plt


def rf_algoritam(X_train, y_train, X_test, y_test, features):

    data_master_RF = RandomForestClassifier(n_estimators=20, random_state=42)
    data_master_RF = data_master_RF.fit(X_train, y_train)
    y_train_pred = data_master_RF.predict(X_train)
    y_test_pred = data_master_RF.predict(X_test)

    # print("Train accuracy RF: ", accuracy_score(y_train, y_train_pred))
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Validation accuracy RandomForest: %.2f%%" % (accuracy * 100.0))

    variance_score = explained_variance_score(y_test, y_test_pred)
    print("Explained variance score RandomForest: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, y_test_pred)
    print("R^2 score RandomForest: %.2f" % r2)

    mean = mean_squared_error(y_test, y_test_pred)
    print("Mean squared error RandomForest: %.2f" % mean)

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


# def slika_stabla_RF(data_master_RF, feature_list):
#     # Pull out one tree from the forest
#     tree = data_master_RF.estimators_[5]
#     # Export the image to a dot file
#     export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
#     # Use dot file to create a graph
#     (graph,) = pydot.graph_from_dot_file('tree.dot')
#     # Write graph to a png file
#     graph.write_png('tree.png')

def feature_importance(forest, features):
    importances = forest.feature_importances_
    indices = np.argsort(importances)

    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()

