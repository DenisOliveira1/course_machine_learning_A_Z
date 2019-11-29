import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#entre [] se por o index das colunas que você quer, é um outro modo
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, -1].values

"""
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

# Avoiding the Dummy Variable Trap
# remove uma das colunas das dummy variables
# a bibliotega de regressão linear do python já faz isso, mas algumas bibliotecas não fazem, então é sempre bom fazer
X = X[:,1:]
"""

# Splitting the dataset in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim

# Naive Bayes

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
cla = GaussianNB()
cla.fit(X_train, Y_train)
"""
import statsmodels.api as sm
cla_OLS = sm.OLS(endog = Y_train, exog = X_train).fit()
cla_OLS.summary()
"""

# Predicting the Test set results
Y_pred = cla.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualising the Train set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, cla.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, cla.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
