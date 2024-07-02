import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

companies = pd.read_csv("1000_Companies.csv")
x = companies.iloc[:, :-1].values
y = companies.iloc[:, -1].values

# print(companies.head())

# sns.heatmap(companies.corr())


lbl_enc = LabelEncoder()
x[:, 3] = lbl_enc.fit_transform(x[:, 3])

# one_hot_enc = OneHotEncoder(categorical_features = [3])
# x = one_hot_enc.fit_transform(x)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(r2_score(y_test, y_pred))