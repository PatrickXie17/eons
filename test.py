import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import funs

boston = load_boston()
bos = pd.DataFrame(boston.data)

bos.columns = boston.feature_names

X = bos
X = preprocessing.scale(X)
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


kw = train_regression(X_train, X_test, y_train, y_test, l2_loss(), thred = 0.01, max_iter = 100, q = 0.01)