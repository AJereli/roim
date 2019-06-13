from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')
# split data train 70 % and test 30 %
data = load_breast_cancer()
# print(data)
x, y, f_name = data['data'], data['target'], data['feature_names']
# print(x[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)


print('Optimal number of features :', rfecv.n_features_)
print('Best features :', f_name[rfecv.support_])

import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


clf_rf = RandomForestClassifier()
clr_rf = clf_rf.fit(x_train,y_train)
importances = clr_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), f_name[indices],rotation=90)
# plt.xlim([-1, x_train.shape[1]])
plt.show()