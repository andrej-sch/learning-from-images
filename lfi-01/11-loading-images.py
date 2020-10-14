import numpy as np
import cv2 as cv

# read image
img_col = cv.imread("./Lenna.png")
# convert to gary scale
img_gray = cv.cvtColor(img_col, cv.COLOR_BGR2GRAY)
img_gray_for_bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

# other way with numpy
# img_gray_for_bgr = np.empty_like(img_col)
# # set values for B channel
# img_gray_for_bgr[:,:, 0] = img_gray
# # set values for G channel
# img_gray_for_bgr[:,:, 1] = img_gray
# # set values for R channel
# img_gray_for_bgr[:,:, 2] = img_gray

# concatinate the images
img = cv.hconcat([img_gray_for_bgr, img_col])
# other ways with numpy
# img = np.hstack((img_gray_for_bgr, img_col))
# img = np.concatenate((img_gray_for_bgr, img_col), axis=1)

cv.imshow('Concatenated image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# inspired by
# https://docs.opencv.org/3.1.0/dc/d2e/tutorial_py_image_display.html
# https://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
# https://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html