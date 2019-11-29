import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# remove uma das colunas das dummy variables
# a bibliotega de regressão linear do python já faz isso, mas algumas bibliotecas não fazem, então é sempre bom fazer
X = X[:,1:]

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

# Multiple Linear Regression

# Fitting Multiple Linear Regression to the Training set
# no fit o algoritmo estuda e aprende a corelação entre as variáveis independentes e a dependente
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = reg.predict(X_test)

# Building the optimal model using Backward Elimination
# adiciona a coluna x0 = 1 para o algoritmo reconehcer o b0 da equação, isso é considerado automaticamente no LinearRegression, mas aqui não. 
# isso é avisado no ctrl+i do OLS
import statsmodels.api as sm
X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()
# remove colunas enquanto o maior p for maior que o limite selecionado, nesse exemplo 0.05, ou seja, 5%
# remove x2
X_opt = X[:,[0,1,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()
# remove x1
X_opt = X[:,[0,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()
# remove x2
X_opt = X[:,[0,3,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()
# remove x2
X_opt = X[:,[0,3]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()

# Predicting the optimal Test set results
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X_opt, Y, test_size = 0.2, random_state = 0)
reg_2 = LinearRegression()
reg_2.fit(X_train_2, Y_train_2)
Y_opt_pred = reg_2.predict(X_test_2)
