import numpy as np
import cv2 as cv
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import path

###############################################################
#                                                             #
# Write your own descriptor / Histogram of Oriented Gradients #
#                                                             #
###############################################################


def plot_histogram(hist: numpy.ndarray, bins: numpy.ndarray, name: str):
    """
    Saves and plots a histogram of angles of gradient vectors.

    Args:
        hist (numpy.ndarray): array of values of the histogram
        bins (numpy.ndarray): array of dtype float of bin edges
        name (str): name of the image, displayed as a plot's title
    """

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.title(name)
    plt.bar(center, hist, align='center', width=width)

    plt.savefig(path.join("./images/hog_test", name + "_hist.png"))
    plt.show()    

def compute_simple_hog(img_color: numpy.ndarray, keypoints: list, name: str) -> numpy.ndarray:
    """
    Computes a histogram of gradients over defined keypoints.

    Args:
        img_color (numpy.ndarray): image in RBG
        keypoints (list): a list of keypoints for which HOG will be calculated
        name (str): name of the image

    Returns:
        numpy.ndarray: an array of descriptors for each keypoint;
                       shape: (number_of_keypoints×number_of_bins)
    """

    # convert color to gray image and extract feature in gray
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

    # compute x and y gradients (sobel kernel size 5)
    sobel_x64f = cv.Sobel(img_gray, cv.CV_64F, dx=1, dy=0, ksize=5)
    sobel_y64f = cv.Sobel(img_gray, cv.CV_64F, dx=0, dy=1, ksize=5)

    # compute magnitude and angle of the gradients
    magnitudes = cv.magnitude(sobel_x64f, sobel_y64f)
    angles = cv.phase(sobel_x64f, sobel_y64f) # in radians

    # go through all keypoints and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        print("Keypoint coordinates: ", kp.pt)
        print("Keypoint size: ", kp.size)
        j, i = np.uint8(kp.pt)
        radius = np.uint8(kp.size/2)
        kp_magnitudes = magnitudes[i-radius:i+radius+1, j-radius:j+radius+1] # shape: (11, 11)
        kp_angles = angles[i-radius:i+radius+1, j-radius:j+radius+1]

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! 
        # Why? Find an answer to that question: 
        # zero magnitude means zero gradients means no change in the color
        # cv.phaze() will return 0 for such zero vectors
        # this will shift the values for the first bin [0, b)
        # therefore we want to exlude such vectors from the histogram
        kp_angles = kp_angles[kp_magnitudes > 0.0]
        (hist, bins) = np.histogram(kp_angles, bins=8, range=(0, 2*np.pi), density=True)

        plot_histogram(hist, bins, name)

        descr[count] = hist

    return descr


keypoints = [cv.KeyPoint(15, 15, 11)]

# test for all test images
diag = cv.imread('./images/hog_test/diag.jpg')
horiz = cv.imread('./images/hog_test/horiz.jpg')
vert = cv.imread('./images/hog_test/vert.jpg')
circle = cv.imread('./images/hog_test/circle.jpg')

images = [diag, horiz, vert, circle]
names = ["diag", "horiz", "vert", "circle"]

descriptors = []
for image, name in zip(images, names):
    descriptors.append(compute_simple_hog(image, keypoints, name))

# inspired by
# https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
# https://docs.opencv.org/master/d3/df2/tutorial_py_basic_ops.html
