import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.reset_index()
print(data)

# split data into features and labels
features = data.iloc[:, 2:44]
labels = data.iloc[:, -1]
print(labels)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=18)
print(X_test)
