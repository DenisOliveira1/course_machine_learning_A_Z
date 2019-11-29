import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Clearning the texts
import re
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(len(dataset["Review"])):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    # com set o for executa mais rápido
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# A seleço regex, lowercar e remoço de stopwords poderiam ser feitas no CountVectiorizer atravez de parametros
# mas não é indicado
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset["Liked"].values #dataset.iloc[:,1].values

# Splitting the dataset in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
"""
# # não precisa de feature scaling, pois só existem 0s e 1s
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# IMPORTANTE: aqui faz só o transform, pois o fit já foi feito no train, isso faz com que ambos esteja escaladas na mesma base
X_test = sc_X.transform(X_test)
# em classificações o Y não precisa ser escalado. Em regreção sim
"""

# Naive Bayes

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
cla = GaussianNB()
cla.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = cla.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_naive_bayes = confusion_matrix(Y_test, Y_pred)








# SVM

# Fitting SVM to the Training set
from sklearn.svm import SVC
cla = SVC(kernel = "linear",
          random_state = 0)
cla.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = cla.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm  = confusion_matrix(Y_test, Y_pred)







# Kernel SVM

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
cla = SVC(kernel = "rbf",
          random_state = 0)
cla.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = cla.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_kernel_svm = confusion_matrix(Y_test, Y_pred)