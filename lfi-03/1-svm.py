from os import path
import glob
import numpy as np
import cv2 as cv
from sklearn import svm


############################################################
#                                                          #
#              Support Vector Machine                      #
#              Image Classification                        #
#                                                          #
############################################################

def create_keypoints(w: int, h: int, size: int=15) -> list:
    """
    Creates keypoints.

    Args:
        w (int): image width
        h (int): image height
        size (int): keypoint diameter

    Returns:
        list: list of generated keypoints
    """

    keypoints = []

    # create a uniform grid
    xv, yv = create_grid(w, h, size=size, dist=size/2, shift=0)

    # create keypoints
    ny, nx = xv.shape
    for i in range(nx):
        for j in range(ny):
            keypoints.append(cv.KeyPoint(xv[j, i], yv[j, i], size))

    return keypoints

def create_grid(w: int, h: int, size: int, dist: int, shift: int) -> tuple:
    """
    Creates a coordinate grid.

    Args:
        w (int): image width
        h (int): image height
        size (int): keypoint diameter
        dist (int): minimum distance between the borders of the two neighboring keypoints
        shift (int): distace between the image edges and keypoint borders at the corners;
                     when shift=0, the first keypoint will be at (size/2, size/2).

    Returns:
        tuple: two ndarrays defining the grid, for x and y axes accordingly
    """

    # number of points along the x axis
    nx = (w - 2*shift) // (size + dist)
    ny = (h - 2*shift) // (size + dist)

    # distribute uniformly
    x = np.linspace(shift+size/2, w-shift-size/2, nx)
    y = np.linspace(shift+size/2, h-shift-size/2, ny)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

    return xv, yv

def compute_descriptor(image: str) -> np.ndarray:
    """
    Calculate the SIFT descriptor.

    Args:
        image (str): path to the image

    Returns:
        np.ndarray: descriptor, array of shape number_of_keypoints×128
    """

    # read the image
    img = cv.imread(image)

    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # calculate the descriptor
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.compute(gray, keypoints)

    return des

def get_class_name(num: int) -> str:
    """
    Gets the name of a class.

    Args:
        num (int): class number

    Returns:
        str: class name
    """

    name = ""
    if num == -1:
        name = 'car'
    elif num == 0:
        name = 'face'
    elif num == 1:
        name = 'flower'

    return name

#---------------------------------------------------------------

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

# create keypoints
keypoints = create_keypoints(256, 256)

# load train images
cars = glob.glob('./images/db/train/cars/*.jpg')
faces = glob.glob('./images/db/train/faces/*.jpg')
flowers = glob.glob('./images/db/train/flowers/*.jpg')

# compute training image descriptors
descriptors = []
for group in (cars, faces, flowers):
    for image in group:
        des = compute_descriptor(image)
        descriptors.append(des)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
ncars = len(cars)
nfaces = len(faces)
nflowers = len(flowers)
train_size = len(descriptors)

# prepare training data
x_train = np.empty((train_size, len(keypoints)*128))
for idx, descriptor in enumerate(descriptors):
    x_train[idx] = np.ndarray.flatten(descriptor)

# set labels
y_train = np.empty(train_size, dtype=np.int8)
y_train[:ncars] = -1 # cars
y_train[ncars:ncars+nfaces] = 0 # faces
y_train[-nflowers:] = 1 # flowers

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.

# clf = svm.SVC(kernel='linear', gamma='scale')
# clf = svm.SVC(kernel='poly', gamma='scale')
# clf = svm.SVC(kernel='sigmoid', gamma='scale')
clf = svm.SVC(gamma="scale") # rbf kernel
clf.fit(x_train, y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

# load test images
test_images = glob.glob('./images/db/test/*.jpg')

# get predictions
results = []
for test_image in test_images:
    des = compute_descriptor(test_image)
    x_test = np.ndarray.flatten(des).reshape(1, -1) # dim: (1, num_keypoints*128)
    y_pred = clf.predict(x_test)
    results.append(y_pred[0])

# 5. output the class + corresponding name
for test_image, result in zip(test_images, results):
    image_name = path.splitext(path.basename(test_image))[0]
    class_name = get_class_name(result)
    print(f'{image_name}: classified as "{class_name}"')

    img = cv.imread(test_image)
    text = f'{class_name.upper()}'
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (255,255,255)
    thickness = 2
    textsize = cv.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - textsize[0]) // 2
    text_y = (img.shape[0] + textsize[1]) // 2
    cv.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv.LINE_AA)
    cv.imshow(image_name, img)

cv.waitKey(0)
cv.destroyAllWindows()


# ispired by:
# https://scikit-learn.org/stable/modules/svm.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
# https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html