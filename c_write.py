import cv2
import matplotlib.pyplot as plt

def simple_color_change(file_name,c1,c2,new_file_name):
	img=cv2.imread(file_name,1)
	img_bgr = cv2.split(img)

	r=img_bgr[2]
	b=img_bgr[0]
	g=img_bgr[1]
	if(c1=='r'):
		if(c2=='g'):
			img_cng = cv2.merge((b,r,g))
		if(c2=='b'):
			img_cng = cv2.merge((r,g,b))
	if(c1=='g'):
		if(c2=='r'):
			img_cng = cv2.merge((b,r,g))
		if(c2=='b'):
			img_cng = cv2.merge((g,b,r))
	if(c1=='b'):
		if(c2=='r'):
			img_cng = cv2.merge((r,g,b))
		if(c2=='g'):
			img_cng = cv2.merge((g,b,r))

	rgb_image=cv2.cvtColor(img_cng,cv2.COLOR_BGR2RGB)
	cv2.imwrite(new_file_name,rgb_image)
	plt.imshow(rgb_image)
	plt.show()
