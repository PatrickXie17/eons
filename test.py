import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import funs

boston = load_boston()
bos = pd.DataFrame(boston.data)

bos.columns = boston.feature_names

X = bos
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

tn = funs.Eons()
kw = tn.train_regression(X_train, X_test, y_train, y_test, funs.Eons.l2_loss(), thred = 0.1, max_iter = 10)