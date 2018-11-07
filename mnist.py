'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import copy

batch_size = 64
num_classes = 10
epochs = 50

center=350
train_num=10000
test_num=15

arr = np.random.rand(train_num)
arr2 = np.random.rand(test_num)
embedded_bit_train=np.round(arr)
embedded_bit_test=np.round(arr2)


# the data, split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

decoded_img = copy.deepcopy(x_test)
encoded_img = copy.deepcopy(x_test)

train_list=[[0 for i in range(10)] for j in range(train_num)]
test_list=[[0 for i in range(10)] for j in range(test_num)]

#mnistに透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。

for y in range(train_num):
    del train_list[y][0:10]
    train_list[y].extend(x_train[y][center-28-1:center-28+2])
    train_list[y].extend(x_train[y][center-1:center+2])
    train_list[y].extend(x_train[y][center+28-1:center+28+2])
    train_list[y].append(embedded_bit_train[y])

for y in range(test_num):
    del test_list[y][0:10]
    test_list[y].extend(x_test[y][center-28-1:center-28+2])
    test_list[y].extend(x_test[y][center-1:center+2])
    test_list[y].extend(x_test[y][center+28-1:center+28+2])
    test_list[y].append(embedded_bit_test[y])

train_list=np.array(train_list)
test_list=np.array(test_list)

y_train = copy.deepcopy(train_list)
y_test = copy.deepcopy(test_list)

y_train = np.delete(y_train, 9, 1)
y_test = np.delete(y_test, 9, 1)




# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(9, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(9, activation='sigmoid'))
model.add(Dropout(0.2))
#model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_list, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_list,y_test))
score = model.evaluate(test_list, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

encoded_data=model.predict(test_list)

import matplotlib.pyplot as plt

encoded_img = copy.deepcopy(x_test)

for i in range(test_num):
    encoded_img[i][center-28-1:center-28+2]=encoded_data[i][0:3]
    encoded_img[i][center-1:center+2]=encoded_data[i][3:6]
    encoded_img[i][center+28-1:center+28+2]=encoded_data[i][6:9]



# 何個表示するか
n = test_num
plt.figure(figsize=(20, 10))
for i in range(n):
    # オリジナルのテスト画像を表示
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(encoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()