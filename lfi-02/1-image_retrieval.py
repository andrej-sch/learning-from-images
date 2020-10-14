import glob
import cv2 as cv
import numpy as np
from os import path
from queue import PriorityQueue

############################################################
#                                                          #
#              Simple Image Retrieval                      #
#                                                          #
############################################################

def create_keypoints(w: int, h: int, size: int=11) -> list:
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

def compute_descriptor(image: str) -> numpy.ndarray:
    """
    Calculate the SIFT descriptor.

    Args:
        image (str): path to the image

    Returns:
        numpy.ndarray: descriptor, array of shape number_of_keypoints×128
    """

    # read the image
    img = cv.imread(image)

    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # calculate the descriptor
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.compute(gray, keypoints)

    return des
        
# implement distance function
def distance(a: numpy.ndaray, b: numpy.ndarray) -> float:
    """
    Computes L2-norm between two descriptors.

    Args:
        a (numpy.ndaray): descriptor 1
        b (numpy.ndaray): descriptor 2

    Returns:
        float: L2-norm
    """

    dist = np.linalg.norm(a-b)

    return dist

def update_max_size(img: numpy.ndarray, max_h: int, max_w: int) -> tuple:
    """
    Returns maximum for height and width.

    Args:
        img (numpy.ndarray): image to compare with.
        max_h (int): current maximum height
        max_w (int): current maximum width

    Returns:
        tubple: updated maximum height and width
    """

    h, w = img.shape[:2]
    max_h = max(max_h, h)
    max_w = max(max_w, w)

    return max_h, max_w

#---------------------------------------------------------------

# 1. preprocessing and load
images = glob.glob('./images/db/train/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
for image in images:
    des = compute_descriptor(image)
    descriptors.append(des)

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())
test_images = glob.glob('./images/db/test/*.jpg')
print(test_images)

queues = []
for test_image in test_images:
    # initialize new queue
    q = PriorityQueue()
    # calculate a descriptor for the test image
    des = compute_descriptor(test_image)
    # add items (distance, image) into the queue
    for descriptor, image in zip(descriptors, images):
        dist = distance(des, descriptor)
        q.put((dist, image))
    # add queue object to the list
    queues.append(q)

# 5. output (save and/or display) the query results in the order of smallest distance
# Format of the output image: test image concatinated with the two rows by 10 of train images.

# At least one image has width/height 255 pixels. The code below implements a save concatination.

# find max height and max width among all train images
max_h, max_w = 0, 0
for image in images:
    img = cv.imread(image)
    max_h, max_w = update_max_size(img, max_h, max_w)

# perform concatination
for test_image, queue in zip(test_images, queues):
    test_img = cv.imread(test_image)
    mh, mw = update_max_size(test_img, max_h, max_w)

    # initialie the output image
    total_h = mh*2
    total_w = mw*11
    output = np.zeros((total_h, total_w, 3), np.uint8)

    test_h, test_w = test_img.shape[:2]
    output[:test_h, :test_w] = test_img

    # first row of train images
    cur_w = test_w
    for i in range(1,11):
        # get image from the queue
        image = queue.get()[1]
        img = cv.imread(image)

        h, w = img.shape[:2]
        output[:h, cur_w:cur_w+w] = img
        cur_w += w

    # second row of train images
    cur_w = test_w
    for i in range(1,11):
        # get image from the queue
        image = queue.get()[1]
        img = cv.imread(image)

        h, w = img.shape[:2]
        output[mh:mh+h, cur_w:cur_w+w] = img
        cur_w += w

    # save the output image
    name = path.basename(test_image)
    cv.imwrite(path.join("./images/db/result", name), output)



# inspired by
# https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
# https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html