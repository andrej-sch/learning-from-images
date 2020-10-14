import cv2 as cv
import numpy as np


# Load image and convert to gray and floating point
img = cv.imread('./images/Lenna.png')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image
sobel_x = cv.Sobel(gray, -1, dx=1, dy=0, ksize=3)
sobel_y = cv.Sobel(gray, -1, dx=0, dy=1, ksize=3)
# -1 -> the destination image will have the same depth as the source

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

Gxx = sobel_x*sobel_x
Gxy = sobel_x*sobel_y
Gyy = sobel_y*sobel_y

window_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
sum_Gxx = cv.filter2D(Gxx, -1, kernel=window_mask)
sum_Gxy = cv.filter2D(Gxy, -1, kernel=window_mask)
sum_Gyy = cv.filter2D(Gyy, -1, kernel=window_mask)
# -1 -> the destination image will have the same depth as the source

# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
# you can compute the resulting image containing the harris corner responses with
# harris = ...

det_M = sum_Gxx*sum_Gyy - sum_Gxy*sum_Gxy
trace_M = sum_Gxx + sum_Gyy
harris = det_M - k*trace_M**2

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use

harris_thres = np.zeros(harris.shape)
harris_thres[harris > threshold*harris.max()] = [255]

# The OpenCV implementation looks like this - please do not change
harris_cv = cv.cornerHarris(gray, 3, 3, k)

# intialize in black - set pixels with corners in white
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv > threshold*harris_cv.max()] = [255]

# just for debugging to create such an image as seen
# in the assignment figure.
img[harris>threshold*harris.max()]=[255,0,0]


# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")


cv.imwrite("Harris_own.png", harris_thres)
cv.imwrite("Harris_cv.png", harris_cv_thres)
cv.imwrite("Image_with_Harris.png", img)

# cv.namedWindow('Interactive Systems: Harris Corner')
# while True:
#     ch = cv.waitKey(0)
#     if ch == 27:
#         break
#
#     cv.imshow('harris',harris_thres)
#     cv.imshow('harris_cv',harris_cv_thres)
