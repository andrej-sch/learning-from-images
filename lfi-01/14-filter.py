import numpy as np
import cv2 as cv


def im2double(img: np.ndarray) -> np.ndarray:
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    Args:
        img (nd.array): an image
    Returns:
        nd.array: normalized image
    """
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size: int, fwhm = 3, center: list = None):
    """
    Makes a square gaussian kernel.

    Args:
        size (int): is the length of a side of the square
        fwhm: is full-width-half-maximum, which
              can be thought of as an effective radius.
        center (list): a center point
    """

    x = np.arange(start=0, stop=size, step=1, dtype=float) # shape: (n,)
    y = x[:,np.newaxis] # shape: (n,1)

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k) # shape: (size, size)


def convolution_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Computes the convolution between a kernel and an image

    Args:
        img (np.ndarray): grayscale image
        kernel (np.array): convolution matrix - 3x3, or 5x5 matrix
    
    Returns: 
        (np.ndarray): result of the convolution
    """

    offset = int(kernel.shape[0]/2)
    newimg = np.zeros(img.shape)

    rows, cols = img.shape
    ksize = kernel.shape[0] # kernel size
    img_padded = np.zeros((rows+2*offset, cols+2*offset))
    img_padded[offset:-offset, offset:-offset] = img

    for i in range(rows):
        for j in range(cols):
            newimg[i, j] = (kernel*img_padded[i:i+ksize, j:j+ksize]).sum()

    return newimg


if __name__ == "__main__":

    # 1. load image in grayscale
    img_path = "./Lenna.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # 2. convert image to 0-1 image (see im2double)
    img = im2double(img)

    # image kernels
    gk = make_gaussian(11)
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # right sobel
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # top sobel

    # 3. use image kernels on normalized image
    img_gb = convolution_2d(img, gk) # gaussian blur
    sobel_x = convolution_2d(img_gb, sobelmask_x)
    sobel_y = convolution_2d(img_gb, sobelmask_y)

    # 4. compute magnitude of gradients
    mog = np.sqrt(sobel_x**2 + sobel_y**2)

    # Show resulting images
    cv.imshow("sobel_x", sobel_x)
    cv.imshow("sobel_y", sobel_y)
    cv.imshow("mog", mog)
    cv.waitKey(0)
    cv.destroyAllWindows()


# some inspirations by:
# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html