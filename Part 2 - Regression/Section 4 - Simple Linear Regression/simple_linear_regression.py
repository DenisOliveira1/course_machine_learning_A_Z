import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# usando :-1 forma uma matrix, usando 0 forma um vetor
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature scaling
# a biblioteca aplica feature scalling internamente automaticamente
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim
"""

# Simple Linear Regression

# Fitting Simple Linear Regression to the Training set
# no fit o algoritmo estuda e aprende a corelação entre as variáveis independentes e a dependente
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = reg.predict(X_test)

# Visualising the Train set results
plt.scatter(X_train,Y_train, color="red")
plt.plot(X_train, reg.predict(X_train), color="blue")
plt.title("Salary x Experience (train set)")
plt.xlabel("Years of Exp￼erience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test,Y_test, color="green")
plt.plot(X_train, reg.predict(X_train), color="blue")
plt.title("Salary x Experience (test set)")
plt.xlabel("Years of Exp￼erience")
plt.ylabel("Salary")
plt.show()

# Visualising the Train and Test set results
plt.scatter(X_train,Y_train, color="red")
plt.scatter(X_test,Y_test, color="green")
plt.plot(X_train, reg.predict(X_train), color="blue")
plt.title("Salary x Experience (train and test sets)")
plt.xlabel("Years of Exp￼erience")
plt.ylabel("Salary")
plt.show()
