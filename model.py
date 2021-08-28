import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\97466\Desktop\week 4\Real estate.csv")
df = pd.DataFrame(data)

X = df[["tran_date","house_age","distance_MRT","no_convenience_stores","latitude","longitude"]]
Y = df[["Target_price"]]

regressor = LinearRegression()
regressor.fit(X,Y)

pickle.dump(regressor, open('model.pkl','wb'))