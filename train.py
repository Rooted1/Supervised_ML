import pandas as pd

data = pd.read_csv('CalMod_06222009.csv')
data = data.dropna()
data = data.reset_index()
print(data)