import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# [linhas, colunas]
# : = todos
# :-1 = todos até ultima coluna (excluindo ela)
# -1 = somente a ultima, nessa tabela -1 seria equivalente à 3
# 1:3 = index 1 e 2
# intervalores assim incluem o primeiro limite e excluem o segundo
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Taking care of missing data
# axis = 0 usa a strategy com as colunas e axis = 1 com as linhas
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# categorical_features já faz referencia que usaremos a coluna de index 0
# esse processo transforma a variável em dummy variables, uma variavel representada por várias colunas, onde cada coluna em 1 representa que esse estado é o ativo
# esse preço deve ser aplicado em variaveis que um estado não assume um valor maior que o outro
# aplicavel á França, Brasil, onde um estado não vale mais que o outro
# não aplicavel á L > M > S, onde um estado vale mais que o outro
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# labelencoder_X não pode ser reusado pois já deu fit em outro parâmetro
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Avoiding the Dummy Variable Trap
# remove uma das colunas das dummy variables
# a bibliotega de regressão linear do python já faz isso, mas algumas bibliotecas não fazem, então é sempre bom fazer
X = X[:,1:]

# Splitting the dataset in train set and test set
# poderia definir o train_size. Os dois juntos é redundante
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
# já que os dados não precisaram mais serem vizualizados é o momento de fazer o feature scaling e "estragar" os dados
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim
