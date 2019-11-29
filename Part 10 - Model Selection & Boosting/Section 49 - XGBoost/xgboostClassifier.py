# https://xgboost.readthedocs.io/en/latest/build.html
# Eu instalei pelo env do anaconda

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1])
X[:,2] = labelencoder_X2.fit_transform(X[:,2])

# categorical_features já faz referencia que usaremos a coluna de index 0
# esse processo transforma a variável em dummy variables, uma variavel representada por várias colunas, onde cada coluna em 1 representa que esse estado é o ativo
# esse processo deve ser aplicado em variaveis que um estado não assume um valor maior que o outro
# aplicavel á França, Brasil, onde um estado não vale mais que o outro
# não aplicavel á L > M > S, onde um estado vale mais que o outro
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# remove uma das colunas das dummy variables
# a bibliotega de regressão linear do python já faz isso, mas algumas bibliotecas não fazem, então é sempre bom fazer
X = X[:,1:] # remove 1 dummy variable de pais
# não precisa fazer essa parte com o genero, pois ele só possui 2 estados (0 e 1)

# Splitting the dataset in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""
# Feature scaling
# decision tree é idependente de feature scalling, ou seja, seu resultado com ou sem featuroing scale é o mesmo
# a biblioteca NÂO aplica feature scalling internamente automaticamente
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim
"""
# XGBoost

# Fitting the a XGBoost to the Training set
# esse import não funciona se esse arquivo se chamar xgboost.py
from xgboost import XGBClassifier
cla = XGBClassifier()
cla.fit(X_train, Y_train)

# Predict the Test set
Y_pred = cla.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
accuracy = (1521 + 208)/2000

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cla,
                             X = X_train,
                             y = Y_train,
                             cv = 10)
accuracies.mean()
accuracies.std()
