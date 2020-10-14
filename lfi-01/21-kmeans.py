import numpy as np
import cv2 as cv
import math
import sys


#############################################################
#                                                           #
#                       KMEANS                              #
#                                                           #
#############################################################

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 2% change rate in the error

def initialize_pos(img: np.ndarray):
    """
    Inittializes the current_cluster_centers array for each cluster 
    with a random pixel position.

    Args:
        img (np.ndarray): input image
    """

    h, w = img.shape[0:2]

    for cluster in range(numclusters):
        i = np.random.randint(h) # row index
        j = np.random.randint(w) # col index
        current_cluster_centers[cluster, 0, :] = img[i, j, :]

    print("Current clusters:\n", current_cluster_centers)

def initialize_dom(img: np.ndarray):
    """
    Inittializes the current_cluster_centers array for each cluster 
    with a random value within the image domain for each channel.

    Args:
        img (np.ndarray): input image
    """

    channels = img.shape[2]

    for cluster in range(numclusters):
        for channel in range(channels):
            cmin = np.amin(img[:,:,channel]) # channel's min
            cmax = np.amax(img[:,:,channel]) # channel's max
            current_cluster_centers[cluster, 0, channel] = np.random.uniform(cmin, cmax)

    print("Current clusters:\n", current_cluster_centers)

def initialize_pp(img: np.ndarray):
    """
    Inittializes the current_cluster_centers array
    according to kmeans++ initialization procedure.

    Args:
        img (np.ndarray): input image
    """

    h, w, c = img.shape
    pixels = img.copy().reshape(h*w, c)

    # Choose one center uniformly at random 
    # from among the data points
    r = np.random.randint(h*w)
    current_cluster_centers[0, 0, :] = pixels[r, :]

    # remove that point from the data set
    pixels = np.delete(pixels, r, axis=0)

    # For each data point x, compute D(x), 
    # the distance between x and the nearest center 
    # that has already been chosen.
    for k in range(1, numclusters):
        dist_sq = np.zeros(pixels.shape[0])
        for i in range(pixels.shape[0]): # over data points
            dist = []
            for j in range(k): # over current clusters
                # calculate distance to the cluster
                diff = pixels[i, :] - current_cluster_centers[j, 0, :]
                dist.append(np.inner(diff, diff))
            
            # choose the distance closest to the cluster
            dist_sq.itemset(i, min(dist))

        probs = dist_sq / dist_sq.sum()
        cumprobs = probs.cumsum()
        r = np.random.uniform()
        for i, prob in enumerate(cumprobs):
            if r <= prob:
                index = i
                break
        
        # add a new cluster
        current_cluster_centers[k, 0, :] = pixels[index, :]

        # remove that point from the data set
        pixels = np.delete(pixels, index, axis=0)


    print("Current clusters:\n", current_cluster_centers)

def update_mean(img: np.ndarray, clustermask: np.ndarray):
    """
    Computes the new cluster centers, i.e. numcluster mean colors
    
    Args:
        img (np.ndarray): input image
        clustermask (np.ndarray): contains the cluster id (int [0..num_clusters])
    """

    for k in range(numclusters):
        current_cluster_centers[k, 0, :] = np.mean(img[clustermask==k], axis=0)

def assign_to_current_mean(img: np.ndarray, clustermask: np.ndarray) -> float:
    """
    Each pixel is assigned to the closest cluster center by updating clustermask.
    
    Args:
        img (np.ndarray): input image
        clustermask (np.ndarray): contains the cluster id for each pixel
                                  (id: int [0...num_clusters))

    Returns:
        float: the overall error (distance) for all pixels to their closest 
               cluster centers (mindistance px - cluster center)
    """

    rows, cols = img.shape[:2]
    distances = np.zeros((numclusters, 1))
    overall_dist = 0

    for i in range(rows):
        for j in range(cols):
            distances = distance(img[i, j, :]) # returned shape: (numclusters, 1)
            
            k = np.argmin(distances) # closest cluster
            clustermask.itemset((i, j), k) # update cluster mask
            overall_dist += distances[k, 0] # sum distance

    return overall_dist


def distance(pixel: np.ndarray) -> np.ndarray:
    '''
    Calculates squared Euclidean distance between a pixel and current cluster centers.

    Args:
        pixel (np.ndarray): a pixel, 1-D array

    Returns:
        ndarray: distances between the pixel and cluster centers
    '''

    # using numpy's broadcasting
    dist = ((pixel-current_cluster_centers)**2).sum(axis=2)

    return dist


def kmeans(img: np.ndarray, max_iter: int = 10, max_change_rate: float = 0.02) -> np.ndarray:
    """
    Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.

    Args:
        img (np.ndarray): input image

    Returns:
        nd.array: color-wise clustered image
    """

    dist = sys.float_info.max

    h, w = image.shape[:2]
    clustermask = np.zeros((h, w), np.uint8)
    result = np.zeros((h, w, 3), np.uint8)

    # initializes cluster centers
    #initialize_pos(img)
    initialize_pp(img)
    overall_dist = assign_to_current_mean(img, clustermask)

    # iterate for a given number of iterations or 
    # if a rate of change is very small
    for step in range(max_iter):
        change = abs(overall_dist-dist) / dist
        if (change < max_change_rate):
            break

        dist = overall_dist
        update_mean(img, clustermask)
        overall_dist = assign_to_current_mean(img, clustermask)

    # update result
    for i in range(h):
        for j in range(w):
            k = clustermask.item((i, j))
            result[i, j, :] = np.array(cluster_colors[k], dtype=np.uint8)

    print("Total within cluster distance: ", round(overall_dist, 2))

    return result

#----------------------------------------------------------------

# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], \
                    [255, 255, 255], [0, 0, 0], [128, 128, 128]]

# number of cluster
numclusters = 3

# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)


# load image
img_path = "./Lenna.png"
imgraw = cv.imread(img_path)
scaling_factor = 0.5
imgraw = cv.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

# read input from the kearboard
while True:
    color_space = int(input("Convert to color space [1(RGB), 2(LAB), 3(HSV)]:"))
    
    if color_space == 1: # stay in RGB
        break
    elif color_space == 2: # convert to LAB
        imgraw = cv.cvtColor(imgraw, cv.COLOR_BGR2LAB)
        break
    elif color_space == 3: # convert to HSV
        imgraw = cv.cvtColor(imgraw, cv.COLOR_BGR2HSV)
        break


# compare different color spaces and their result for clustering
image = imgraw

# k-means over the image
# returns a result image where each pixel is a color with one of the 
# cluster_colors depending on its cluster assignment
res = kmeans(image)

# concatinate quantized and original images
h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1+w2] = image

cv.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv.waitKey(0)
cv.destroyAllWindows()


# some inspirations by
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
# https://en.wikipedia.org/wiki/K-means%2B%2B
# http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
# https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm