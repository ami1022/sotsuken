import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
 
DataDir = 'sample1'

imgFile = cv2.imread('m17.png')
resizeImg = cv2.resize(imgFile,(480,480))
rgbImg = cv2.cvtColor(resizeImg,cv2.COLOR_BGR2RGB)
plt.imshow(rgbImg)
plt.show()
 
ImgB,ImgG,ImgR = cv2.split(resizeImg) #BGRを分離する
detector = cv2.AKAZE_create()
    
keypoints,desc = detector.detectAndCompute(ImgB,None)#特徴量の計算
out = cv2.drawKeypoints(rgbImg, keypoints, None)
print('特徴点の数は '+str(len(keypoints)))
print('descripterの次元は'+str(desc.shape))
print(desc[10])
plt.plot(desc[150])
plt.show()
plt.imshow(out)
plt.show()
