import pdb
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt

# load image
img = cv2.imread('0.jpg')

# compute laplacian
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
laplacian = cv2.Laplacian(gray,cv2.CV_64F)

# normalize laplacian in between 0 and 1
max_val = laplacian.max()
min_val = laplacian.min()
laplacian_norm = laplacian/(max_val-min_val)
laplacian_norm += abs(laplacian_norm.min())

# show laplacian
plt.figure()
plt.imshow(laplacian_norm, cmap='gray')



