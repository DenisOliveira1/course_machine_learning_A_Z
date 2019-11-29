import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Instalando o Keras já instala o Tensorflow e o Theano

# Part 1 - Data Preprocessing

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

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim

# ANN

# Part 2 - Making the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
cla = Sequential()

# Adding the input layer and the first hidden layer
# usa-se na hidden layer a media entre a input layer e a output layer: (11 + 1) / 2 = 6
# output layer será somente 0 ou 1, logo apenas 1 node na output layer
cla.add(Dense(output_dim = 6,
              init = 'uniform',
              activation = "relu",
              input_dim = 11))
# Adding the second hidden layer
cla.add(Dense(output_dim = 6,
              init = 'uniform',
              activation = "relu"))

# Adding the output layer
# se houvesse mais que duas categorias o unit seria atualizado de acordo com o numero
# de dummy variables e a activation seria a softmax (sigmoid para mais que dois estados)
cla.add(Dense(output_dim = 1,
              init = 'uniform',
              activation = "sigmoid"))

# Compiling the ANN
# se houvesse mais que duas categorias a função de loss seria
# categorical_crossentropy 
cla.compile(optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["accuracy"])

# Fitting the ANN to the training set
# batch_size é depois de quantas rows que os weights serão atualizados
# é o número de epocas ou seja, o número de vezes que o dataset inteiro será usado
cla.fit(X_train,
        Y_train,
        batch_size = 10,
        epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the test set results
Y_prob = cla.predict(X_test)
"""
Y_pred = []
for i in Y_prob:
    if i > 0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)
"""
Y_pred = (Y_prob > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])