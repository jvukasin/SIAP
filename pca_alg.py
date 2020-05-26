from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import random_forest_alg as rf
import xgboost_alg as xgb


def pca_algoritam(x_train, y_train, x_test, y_test, features):

    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # print('y_test:')
    # print(y_test)
    # y = y_test.reshape(-1, 1)
    # y = RobustScaler(quantile_range=(25, 75)).transform(y)
    # print('y:')
    # print(y)

    # Make an instance of the Model
    pca = PCA(.95)
    # print('shape: ', x_train.shape)
    # pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(x_train)

    print('components: ', pca.n_components_)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    # features = x_train.columns
    # print('x_test')
    # print(x_test)

    """# RandomForest algoritam"""
    print('===============rf - pca===============')
    rf.rf_algoritam(x_train, y_train, x_test, y_test, features)

    """# XGBoost algoritam"""
    # xgb.xbg_algoritam(x_train, y_train, x_test, y_test)
