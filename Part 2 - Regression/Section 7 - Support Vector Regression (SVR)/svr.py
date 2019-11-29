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
# a biblioteca NÂO aplica feature scalling internamente automaticamente
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# SVR

# Fitting SVR to the dataset
from sklearn.svm import SVR
reg = SVR(kernel = "rbf")
reg.fit(X,Y)

# Prediction a new result with SVR
# a escala do problema mudou por causa do feature scaling, logo a entrada x deve ser sofre o mesmo processo
x = sc_X.transform(np.array(6.5).reshape(-1,1))
Y_pred = reg.predict(x)
# a previsão é feita no novo sistema de escala, mas queremos vê-la em sua escala original
Y_pred = sc_Y.inverse_transform(Y_pred)

# Visualising the Regression results
plt.scatter(X,Y, color = "red")
plt.plot(X, reg.predict(X), color = "blue")
plt.scatter(x, reg.predict(x), color = "green")
plt.title("SVR")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR results (grid)
X_grid = np.arange(min(X),max(X), step = 0.1)# vetor
X_grid = X_grid.reshape((len(X_grid),1))# matriz

plt.scatter(X,Y, color = "red")
plt.plot(X_grid, reg.predict(X_grid), color = "blue")
plt.scatter(x, reg.predict(x), color = "green")
plt.title("SVR (grid = 0.1)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

