import cv2
img = cv2.imread('LENNA.png')
img = img.astype('float32') / 255.
img = img.reshape((len(img), np.prod(img.shape[1:])))