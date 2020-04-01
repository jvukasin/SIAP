from sklearn.ensemble import RandomForestClassifier #RF klasifikator
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.tree import export_graphviz
import pydot


def rf_algoritam(X_train, y_train, X_test, y_test):

    data_master_RF = RandomForestClassifier(n_estimators=50, random_state=42)
    data_master_RF = data_master_RF.fit(X_train, y_train)
    y_train_pred = data_master_RF.predict(X_train)
    y_test_pred = data_master_RF.predict(X_test)

    print("Train accuracy RF: ", accuracy_score(y_train, y_train_pred))
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Validation accuracy RandomForest: %.2f%%" % (accuracy * 100.0))

    variance_score = explained_variance_score(y_test, y_test_pred)
    print("Explained variance score RandomForest: %.2f%%" % (variance_score * 100.0))
    # slika_stabla_RF(data_master_RF, feature_list)


# def slika_stabla_RF(data_master_RF, feature_list):
#     # Pull out one tree from the forest
#     tree = data_master_RF.estimators_[5]
#     # Export the image to a dot file
#     export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
#     # Use dot file to create a graph
#     (graph,) = pydot.graph_from_dot_file('tree.dot')
#     # Write graph to a png file
#     graph.write_png('tree.png')
