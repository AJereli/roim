from sklearn.datasets import load_boston
import pandas as pd
from sklearn import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

boston_ds = load_boston()

boston_dataset = boston_ds.data

mean = boston_dataset.mean(axis=0)
boston_dataset -= mean
std = boston_dataset.std(axis=0)
boston_dataset /= std

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
X, y = shuffle(boston_dataset, boston_ds.target, random_state=13)

clf = LassoCV(cv=5)

sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

plt.title("Features selected from Boston using SelectFromModel with ""threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
#plt.show()

important_X = [list(t) for t in list(zip(feature1, feature2))]
print((important_X[1]))
X, y = shuffle(important_X, boston_ds.target, random_state=13)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
skm = LinearRegression()

skm.fit(X_train, y_train)
mse = mean_squared_error(y_test, skm.predict(X_test))

Y_pred = skm.predict(X_test)
print("MSE: %.4f" % mse)

plt.scatter(y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
