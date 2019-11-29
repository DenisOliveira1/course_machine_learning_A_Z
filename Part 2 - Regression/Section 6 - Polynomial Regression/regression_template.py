import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

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

# Regression

# Fitting Regression to the dataset

# Prediction a new reuslt with Regression
Y_pred = reg.predict(np.array(6.5).reshape(-1,1))

# Visualising the Regression results
plt.scatter(X,Y, color = "red")
plt.plot(X, reg.predict(X), color = "blue")
plt.title("Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Regression results (grid)
X_grid = np.arange(min(X),max(X), step = 0.1)# vetor
X_grid = X_grid.reshape((len(X_grid),1))# matriz

plt.scatter(X,Y, color = "red")
plt.plot(X_grid, reg.predict(X_grid), color = "blue")
plt.title("Regression (grid = 0.1)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

