import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
#coloquei : no y para virar uma matrix de uma coluna, pois é requisito para o feature scalling
Y = dataset.iloc[:, -1:].values

# Splitting the dataset in train set and test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""

# Feature scaling
# decision tree é idependente de feature scalling, ou seja, seu resultado com ou sem featuroing scale é o mesmo
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim
"""

# Decision Tree Regression

# Fitting Decision Tree Regression to the dataset
# é um modelo não continuo pois divide sua decisão equivalente a quantidade de folhas da arvvore de decisão
# é um modelo interessante para problemas multidimenssionais
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X,Y)

# Prediction a new result with Decision Tree Regression
x = np.array(6.5).reshape(-1,1)
Y_pred = reg.predict(x)

# Visualising the Decision Tree Regression results
plt.scatter(X,Y, color = "red")
plt.plot(X, reg.predict(X), color = "blue")
plt.scatter(x, reg.predict(x), color = "green")
plt.title("Decision Tree Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Decision Tree Regression results (grid)
X_grid = np.arange(min(X),max(X), step = 0.01)# vetor
X_grid = X_grid.reshape((len(X_grid),1))# matriz

plt.scatter(X,Y, color = "red")
plt.plot(X_grid, reg.predict(X_grid), color = "blue")
plt.scatter(x, reg.predict(x), color = "green")
plt.title("Decision Tree Regression (grid = 0.01)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

