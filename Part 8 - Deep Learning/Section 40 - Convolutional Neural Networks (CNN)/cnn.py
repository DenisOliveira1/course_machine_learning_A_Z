import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Instalando o Keras já instala o Tensorflow e o Theano

# CNN

# Part 1 - Making the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
cla = Sequential()

# Step 1 - Convolution
cla.add(Convolution2D(filters = 32, # numero de filtros
                      kernel_size = [3,3],# dimensão da convolução
                      strides = (1,1), # passo da convolução
                      input_shape = (64, 64, 3), # canais de cores
                      activation = "relu")) # remover linearidade da convolução

# Step 2 - Max Pooling
# Reduz a dimensionalidade não sobrecarregando os dados que serão entregues à ANN
# Mantem a informação principal, evita overfitting e diminui a necessidade de processamento
cla.add(MaxPool2D(pool_size = (2,2)))

# Adding second convolution layer
cla.add(Convolution2D(filters = 32, # numero de filtros
                      kernel_size = [3,3],# dimensão da convolução
                      strides = (1,1), # passo da convolução
                      activation = "relu")) # remover linearidade da convolução
cla.add(MaxPool2D(pool_size = (2,2)))

# Step 3 - Flattening
# forma vetores com as saidas do Max Pooling
cla.add(Flatten())

# Step 4 - Full connection
# Adding the hidden layer
# por experimentação ele indica 128
cla.add(Dense(output_dim = 128,
              init = 'uniform',
              activation = "relu"))

# Adding the output layer
# se houvesse mais que duas categorias o unit seria atualizado de acordo com o numero
# de dummy variables e a activation seria a softmax (sigmoid para mais que dois estados)
cla.add(Dense(output_dim = 1,
              init = 'uniform',
              activation = "sigmoid"))

# Compiling the CNN
# se houvesse mais que duas categorias a função de loss seria
# categorical_crossentropy 
cla.compile(optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["accuracy"])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),# mesma dimenssao dos canais escolhidos na CNN
        batch_size=32,
        class_mode='binary')# variavel dependente

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),# mesma dimenssao dos canais escolhidos na CNN
        batch_size=32,
        class_mode='binary')# variavel dependente

cla.fit_generator(
        train_set,
        steps_per_epoch= 8000, # número de instancias no train_set
        epochs=25,
        validation_data = test_set,
        validation_steps = 2000) # número de imagens no test_set
