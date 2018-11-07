import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

#教師データ整形、とりあえず100枚の画像の同じ箇所に透かしビット埋め込み。

center=350
embedded_bit=1
list=[[0 for i in range(10)] for j in range(100)]
for y in range(100):
	list[y].extend(x_train[y][center-28-1:center-28+2])
	list[y].extend(x_train[y][center-1:center+2])
	list[y].extend(x_train[y][center+28-1:center+28+2])
	list[y].append(embedded_bit)

print(list)


train_list=[[0 for i in range(10)] for j in range(100)]
test_list=[[0 for i in range(10)] for j in range(10)]

for y in range(100):
    train_list[y].extend(x_train[y][center-28-1:center-28+2])
    train_list[y].extend(x_train[y][center-1:center+2])
    train_list[y].extend(x_train[y][center+28-1:center+28+2])
    train_list[y].append(embedded_bit)