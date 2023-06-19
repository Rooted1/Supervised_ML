import pandas as pd

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.reset_index()
print(data)

# split data into features and labels
features = data.iloc[:, 2:44]
labels = data.iloc[:, -1]
print(labels)