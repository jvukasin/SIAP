from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random_forest_alg as rf
import xgboost_alg as xgb


def pca_algoritam(x_train, y_train, x_test, y_test):

    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Make an instance of the Model
    pca = PCA(.95)
    pca.fit(x_train)

    print('components: ', pca.n_components_)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    """# RandomForest algoritam"""
    print('===============rf - pca===============')
    rf.rf_algoritam(x_train, y_train, x_test, y_test)

    """# XGBoost algoritam"""
    # xgb.xbg_algoritam(x_train, y_train, x_test, y_test)
