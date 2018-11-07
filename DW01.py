import cv2
import matplotlib.pyplot as plt
import numpy as np

def simple_read_bw(file_name,new_file_name):
	img=cv2.imread(file_name)
	return img

img=simple_read_bw("images/LENNA.png","Cover.png")

#plt.imshow(img)
#plt.show()
x=0
va=dict()
for i in img:
	value=i[0]
	value=value[0]
	va[value]=
	print(value)


