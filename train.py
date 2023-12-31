import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.drop(['id', 'Unnamed: 0'], axis=1)
data = data.reset_index()
print(data)

# split data into features and labels
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=10)
print(X_test)

# train and evaluate model using Random forest classifier and make predictions
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy on test data is %.4f' % (accuracy_score(y_test, y_pred)))

# use sklearn PCA to improve accuracy
pca = PCA().fit(features)
# print(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# plt.show()

# specify components to improve results
pca = PCA(n_components=2).fit(features)
X_pca = pca.transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.3, random_state=10)

# fit training model and predict performance
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy on new test data is %.4f' % (accuracy_score(y_test, y_pred)))