from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model
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
decoded_epochs = 200

#ステゴ画像生成ニューラルネットワークモデル生成

#autoencoderモデル生成
#encoderのモデル生成
encoding_dim_e = 9  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img_e = Input(shape=(10,))
encoded_e = Dense(encoding_dim_e, activation='sigmoid')(input_img_e)
decoded_e = Dense(10, activation='sigmoid')(encoded_e)

autoencoder_e = Model(input_img_e, decoded_e)
encoder_e = Model(input_img_e, encoded_e)
encoded_input_e = Input(shape=(encoding_dim_e,))
decoder_layer_e = autoencoder_e.layers[-1]
decoder_e = Model(encoded_input_e, decoder_layer_e(encoded_input_e))

autoencoder_e.compile(optimizer='adam', loss='binary_crossentropy')

model_json_str = autoencoder_e.to_json()
open('encode.json', 'w').write(model_json_str)


#decoderのモデル生成
encoding_dim_d = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img_d = Input(shape=(9,))
encoded_d = Dense(encoding_dim_d, activation='sigmoid')(input_img_d)
decoded_d = Dense(10, activation='sigmoid')(encoded_d)

autoencoder_d = Model(input_img_d, decoded_d)
encoder_d = Model(input_img_d, encoded_d)
encoded_input_d = Input(shape=(encoding_dim_d,))
decoder_layer_d = autoencoder_d.layers[-1]
decoder_d = Model(encoded_input_d, decoder_layer_d(encoded_input_d))

autoencoder_d.compile(optimizer='adam', loss='binary_crossentropy')

model_json_str = autoencoder_d.to_json()


#入力データ形成
#透かしビット系列作成
arr = np.random.rand(train_num)
arr2 = np.random.rand(test_num)
arr=arr.reshape(len(arr),1)
arr2=arr2.reshape(len(arr2),1)
embedded_bit_train=np.round(arr)
embedded_bit_test=np.round(arr2)

#データダウンロード
(x_train, _), (_ , _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

#テストデータ用の画像読み込み
x_test=cv2.imread("sea.jpg")
x_test=cv2.cvtColor(x_test,cv2.COLOR_BGR2RGB).astype('float32')/255


#後で画像表示する用
decoded_img = copy.deepcopy(x_test)
encoded_img = copy.deepcopy(x_test)

#透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。
#ステゴ画像作るときのラベルがわり
y_train_r=copy.deepcopy(x_train[0:train_num,20:23,20:23,0:1]).reshape(train_num,3,3).reshape(train_num,9)
y_train_g=copy.deepcopy(x_train[0:train_num,20:23,20:23,1:2]).reshape(train_num,3,3).reshape(train_num,9)
y_train_b=copy.deepcopy(x_train[0:train_num,20:23,20:23,2:3]).reshape(train_num,3,3).reshape(train_num,9)
y_test_r=copy.deepcopy(x_test[2000:2003,2000:2003,0:1]).reshape(test_num,3,3).reshape(test_num,9)
y_test_g=copy.deepcopy(x_test[2000:2003,2000:2003,1:2]).reshape(test_num,3,3).reshape(test_num,9)
y_test_b=copy.deepcopy(x_test[2000:2003,2000:2003,2:3]).reshape(test_num,3,3).reshape(test_num,9)


#画素値に透かし情報足して入力データ形成

train_list_r=copy.deepcopy(y_train_r)
train_list_g=copy.deepcopy(y_train_g)
train_list_b=copy.deepcopy(y_train_b)
test_list_r=copy.deepcopy(y_test_r)
test_list_g=copy.deepcopy(y_test_g)
test_list_b=copy.deepcopy(y_test_b)

train_list_r = np.hstack((train_list_r,embedded_bit_train))
train_list_g = np.hstack((train_list_g,embedded_bit_train))
train_list_b = np.hstack((train_list_b,embedded_bit_train))
test_list_r = np.hstack((test_list_r,embedded_bit_test))
test_list_g = np.hstack((test_list_g,embedded_bit_test))
test_list_b = np.hstack((test_list_b,embedded_bit_test))


print(train_list_b[3])



#いざ学習
#原画像入れてステゴ画像出す

#autoencoder
autoencoder_e.fit(train_list_r, train_list_r,
                nb_epoch=encoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_list_r, test_list_r))

autoencoder_e.fit(train_list_g, train_list_g,
                nb_epoch=encoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_list_g, test_list_g))

autoencoder_e.fit(train_list_b, train_list_b,
                nb_epoch=encoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_list_b, test_list_b))

rate_o=0.3
rate_y=0.7

output_r = copy.deepcopy(encoder_e.predict(train_list_r))
train_data2_r=output_r*rate_o+y_train_r*rate_y

output_g = copy.deepcopy(encoder_e.predict(train_list_g))
train_data2_g=output_g*rate_o+y_train_g*rate_y

output_b=copy.deepcopy(encoder_e.predict(train_list_b))
train_data2_b=output_b*rate_o+y_train_b*rate_y

output2_r=encoder_e.predict(test_list_r)
test_data2_r=output2_r*rate_o+y_test_r*rate_y

output2_g=encoder_e.predict(test_list_g)
test_data2_g=output2_g*rate_o+y_test_g*rate_y

output2_b=encoder_e.predict(test_list_b)
test_data2_b=output2_b*rate_o+y_test_b*rate_y


autoencoder_e.save_weights('encode.h5')

autoencoder_d.fit(train_data2_r, train_list_r,
                epochs=decoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_data2_r, test_list_r))

autoencoder_d.fit(train_data2_g, train_list_g,
                epochs=decoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_data2_g, test_list_g))

autoencoder_d.fit(train_data2_b, train_list_b,
                epochs=decoded_epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(test_data2_b, test_list_b))

encoded_data_r = encoder_d.predict(test_data2_r)
decoded_data_r = decoder_d.predict(encoded_data_r)

encoded_data_g = encoder_d.predict(test_data2_g)
decoded_data_g = decoder_d.predict(encoded_data_g)

encoded_data_b = encoder_d.predict(test_data2_b)
decoded_data_b = decoder_d.predict(encoded_data_b)



print(embedded_bit_test)
print(decoded_data_g)

print(len(test_data2_r[0]))

autoencoder_d.save_weights('decode.h5')
"""
test_data2_r=np.delete(test_data2_r,9,1)
test_data2_g=np.delete(test_data2_g,9,1)
test_data2_b=np.delete(test_data2_b,9,1)
"""
decoded_data_r=np.delete(decoded_data_r,9,1)
decoded_data_g=np.delete(decoded_data_g,9,1)
decoded_data_b=np.delete(decoded_data_b,9,1)


#類似度計算
result1=np.sqrt(sum(np.square(test_data2_r[0]*255-y_test_r[0]*255)))
result2=np.sqrt(sum(np.square(decoded_data_r[0]*255-y_test_r[0]*255)))
print(test_data2_r,y_test_r)
print(1/(1 + result1))
print(1/(1 + result2))


encoded_img[2000:2003,2000:2003,0:1]=test_data2_r.reshape(test_num,3,3,1)
encoded_img[2000:2003,2000:2003,1:2]=test_data2_g.reshape(test_num,3,3,1)
encoded_img[2000:2003,2000:2003,2:3]=test_data2_b.reshape(test_num,3,3,1)

decoded_img[2000:2003,2000:2003,0:1]=decoded_data_r.reshape(3,3,1)
decoded_img[2000:2003,2000:2003,1:2]=decoded_data_g.reshape(3,3,1)
decoded_img[2000:2003,2000:2003,2:3]=decoded_data_b.reshape(3,3,1)



#画像表示
n = test_num
i=0
plt.figure(figsize=(20, 5))
# オリジナルのテスト画像を表示
ax = plt.subplot(1, 3, i+1)
plt.imshow(x_test)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 3, i+1+n)
plt.imshow(encoded_img)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# 変換された画像を表示
ax = plt.subplot(1, 3, i+1+(2*n))
plt.imshow(decoded_img)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

encoded_img=cv2.cvtColor(encoded_img,cv2.COLOR_BGR2RGB)*255
cv2.imwrite("sea_test.png",encoded_img)


t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")

