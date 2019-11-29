import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset in train set and test set
# o dataset é muito pequeno e precisamos de toda informação para fazer as melhores previsões possiveis
# por isso não haverá split
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""

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

# Polynomial Regression

# Fitting Linear Regression to the dataset just for comparing
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# se você por o degree 3 vão criadas coulas 2 e 3
# essa função cria a coluna de x0 = 1 para o algoritimo não ignorar b0, igual na Multiple Linear Regression
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Linear Regression results 
plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.scatter(6.5, lin_reg.predict(np.array(6.5).reshape(-1,1)), color = "green")
plt.title("Simple Linear Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (degree = 2)
plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Polynomial Regression (degree = 2)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (degree = 4)
# quando maior o degree mais preciso a previsão é, porém pode haver overfitting
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Polynomial Regression (degree = 4)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (degree = 4 + grid)
# plotar um grid com mais pontos traz uma melhor suavização ao gráfico
X_grid = np.arange(min(X),max(X), step = 0.1)# vetor
X_grid = X_grid.reshape((len(X_grid),1))# matriz

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X,Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.scatter(6.5, lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1,1))), color = "green")
plt.title("Polynomial Regression (degree = 4) + (grid = 0.1)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Prediction a new reuslt with Linear Regression
pred_lin = lin_reg.predict(np.array(6.5).reshape(-1,1))

# Prediction a new reuslt with Polyminal Regression (degree = 4 + grid)
pred2_pol = lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1,1)))
