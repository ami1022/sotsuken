import cv2
import matplotlib.pyplot as plt

def simple_read_bw(file_name,new_file_name):
	img=cv2.imread(file_name)
	plt.imshow(img)
	cv2.imwrite(new_file_name,img)
	plt.show()

import sys

args = sys.argv
simple_read_bw("/Users/nishizakiami/program/sotsuken/images/"+args[1],args[2])