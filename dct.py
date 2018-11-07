#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

"""
参考:
[1]『画像処理とパターン認識入門』酒井幸市 著
[2] scipy.fftpack.dct http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
"""

class DCT:
	def __init__(self,N):
		self.N = N	# データ数．
		# 1次元，2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
		self.phi_1d = np.array([ self.phi(i) for i in range(self.N) ])
		
		# Nが大きいとメモリリークを起こすので注意
		# MNISTの28x28程度なら問題ない
		self.phi_2d = np.zeros((N,N,N,N))
		for i in range(N):
			for j in range(N):
				phi_i,phi_j = np.meshgrid(self.phi_1d[i],self.phi_1d[j])
				self.phi_2d[i,j] = phi_i*phi_j

	def dct(self,data):
		""" 1次元離散コサイン変換を行う """
		return self.phi_1d.dot(data)
	
	def idct(self,c):
		""" 1次元離散コサイン逆変換を行う """
		return np.sum( self.phi_1d.T * c ,axis=1)
	
	def dct2(self,data):
		""" 2次元離散コサイン変換を行う """
		return np.sum(self.phi_2d.reshape(N*N,N*N)*data.reshape(N*N),axis=1).reshape(N,N)
	
	def idct2(self,c):
		""" 2次元離散コサイン逆変換を行う """
		return np.sum((c.reshape(N,N,1)*self.phi_2d.reshape(N,N,N*N)).reshape(N*N,N*N),axis=0).reshape(N,N)
	
	def phi(self,k):
		""" 離散コサイン変換(DCT)の基底関数 """
		# DCT-II
		if k == 0:
			return np.ones(self.N)/np.sqrt(self.N)
		else:
			return np.sqrt(2.0/self.N)*np.cos((k*np.pi/(2*self.N))*(np.arange(self.N)*2+1))
		# DCT-IV(試しに実装してみた)
		#return np.sqrt(2.0/N)*np.cos((np.pi*(k+0.5)/self.N)*(np.arange(self.N)+0.5))

if __name__=="__main__":
	N = 28			# データの次元は10x10とする
	dct = DCT(28)	# 離散コサイン変換を行うクラスを作成
	# サンプル画像を作る
	img=Image.open("m17.png")
	img= img.convert('L').resize((28,28),Image.LANCZOS)
	contrast_converter = ImageEnhance.Contrast(img)
	contrast_img = contrast_converter.enhance(6.0)

	img_array = np.array(contrast_img)
	
	c = dct.dct2(img_array)	# 2次元離散コサイン変換
	y = dct.idct2(c)	# 2次元離散コサイン逆変換
	
	# 元の画像と復元したものを表示
	plt.subplot(1,2,1)
	plt.imshow(img,cmap="Greys")
	plt.title("original")
	plt.xticks([])
	plt.subplot(1,2,2)
	plt.imshow(y,cmap="Greys")
	plt.title("restored")
	plt.xticks([])
	plt.show()