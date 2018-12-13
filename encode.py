from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_json
from keras.datasets import cifar10

import numpy as np
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import copy

import matplotlib.pyplot as plt

import time

t1 = time.time()

center=16
train_num=500
test_num=1

batch_size = 128
encoded_epochs = 200
decoded_epochs = 100

#ステゴ画像生成ニューラルネットワークモデル生成

#autoencoderモデル生成
encoding_dim = 9  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img = Input(shape=(10,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(10, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


#モデル、重み読み込み
#autoencoder = model_from_json(open('encode.json').read())
autoencoder.load_weights('encode.h5')

#入力データ形成
#透かしビット系列作成
arr = np.random.rand(test_num)
arr=arr.reshape(len(arr),1)
embedded_bit_test=np.round(arr)

print("埋め込みビット",embedded_bit_test)


#x_test = x_test.astype('float32') / 255.

#テストデータ用の画像読み込み
x_test=cv2.imread("sea.jpg")
x_test=cv2.cvtColor(x_test,cv2.COLOR_BGR2RGB).astype('float32')/255

#後で画像表示する用
encoded_img = copy.deepcopy(x_test)

#透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。
#ステゴ画像作るときのラベルがわり
y_test_r=copy.deepcopy(x_test[2000:2003,2000:2003,0:1]).reshape(test_num,3,3).reshape(test_num,9)
y_test_g=copy.deepcopy(x_test[2000:2003,2000:2003,1:2]).reshape(test_num,3,3).reshape(test_num,9)
y_test_b=copy.deepcopy(x_test[2000:2003,2000:2003,2:3]).reshape(test_num,3,3).reshape(test_num,9)

#画素値に透かし情報足して入力データ形成
test_list_r=copy.deepcopy(y_test_r)
test_list_g=copy.deepcopy(y_test_g)
test_list_b=copy.deepcopy(y_test_b)

test_list_r = np.hstack((test_list_r,embedded_bit_test))
test_list_g = np.hstack((test_list_g,embedded_bit_test))
test_list_b = np.hstack((test_list_b,embedded_bit_test))



#いざ学習
#原画像入れてステゴ画像出す

#autoencoder
autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


output_r=encoder.predict(test_list_r)
output_g=encoder.predict(test_list_g)
output_b=encoder.predict(test_list_b)

output_r=output_r*0.3+y_test_r*0.7
output_g=output_g*0.3+y_test_g*0.7
output_b=output_b*0.3+y_test_b*0.7


result1=np.sqrt(sum(np.square(output_r[0]*255-y_test_r[0]*255)))
print(1/(1 + result1))

encoded_img[2000:2003,2000:2003,0:1]=output_r.reshape(test_num,3,3,1)
encoded_img[2000:2003,2000:2003,1:2]=output_g.reshape(test_num,3,3,1)
encoded_img[2000:2003,2000:2003,2:3]=output_b.reshape(test_num,3,3,1)



#画像表示
n = test_num
i=0
plt.figure(figsize=(8, 5))
# オリジナルのテスト画像を表示
ax = plt.subplot(1, 3, i+1)
plt.imshow(x_test)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 3, i+1+n)
plt.imshow(encoded_img)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

encoded_img=cv2.cvtColor(encoded_img,cv2.COLOR_BGR2RGB)*255
cv2.imwrite("sea_stego.png",encoded_img)

t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")