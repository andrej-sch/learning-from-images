import numpy as np
import cv2 as cv

def update_mode(mode: int) -> int:

    # wait for key and switch to mode
    ch = cv.waitKey(1) & 0xFF

    if ch == ord('1'): # change color space to HSV
        mode = 1
    elif ch == ord('2'): # change color space to LAB
        mode = 2
    elif ch == ord('3'): # change color space to YUV
        mode = 3
    elif ch == ord('4'): # Gaussian adaptive thresholding
        mode = 4
    elif ch == ord('5'): # Otsu's thresholding
        mode = 5
    elif ch == ord('6'): # Otsu's thresholding after Gaussian filtering
        mode = 6
    elif ch == ord('7'): # Canny edge extraction
        mode = 7
    elif ch == ord('0'): # reset
         mode = 0
    elif ch == ord('q'): # quit
        mode = -1

    return mode

def apply_mode(mode: int, frame: np.ndarray) -> np.ndarray:
    
    font = cv.FONT_HERSHEY_SIMPLEX # for captures

    if mode == 1:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        text = 'HSV color space'
        cv.putText(frame, text, (10,20), font, 0.8, (0,0,0), 2, cv.LINE_AA)
    elif mode == 2:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

        text = 'LAB color space'
        cv.putText(frame, text, (10,20), font, 0.8, (0,0,0), 2, cv.LINE_AA)
    elif mode == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2YUV)

        text = 'YUV color space'
        cv.putText(frame, text, (10,20), font, 0.8, (0,0,0), 2, cv.LINE_AA)
    elif mode == 4:
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv.THRESH_BINARY, blockSize=11, C=2)

        text = 'Gaussian adaptive thresholding'
        cv.putText(frame, text, (10,20), font, 0.8, (0,0,0), 2, cv.LINE_AA)
    elif mode == 5:
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Otsu's thresholding
        ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        text = 'Otsu\'s thresholding'
        cv.putText(frame, text, (10,20), font, 0.8, (255,255,255), 2, cv.LINE_AA)
    elif mode == 6:
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Gaussian filtering
        frame = cv.GaussianBlur(frame, (5,5), 0)
        # Otsu's thresholding
        ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        text = 'Gaussian filtering + Otsu\'s thresholding'
        cv.putText(frame, text, (10,20), font, 0.8, (255,255,255), 2, cv.LINE_AA)
    elif mode == 7:
        frame = cv.Canny(frame, 100, 200)

        text = 'Canny edge extraction'
        cv.putText(frame, text, (10,20), font, 0.8, (255,255,255), 2, cv.LINE_AA)

    return frame


if __name__ == "__main__":
    
    cap = cv.VideoCapture(0)
    mode = 0

    while(True):
        # capture frame-by-frame
        ret, frame = cap.read()

        mode = update_mode(mode)
        if mode == -1:
            break
        else:
            frame = apply_mode(mode, frame)

        # Display the resulting frame
        cv.imshow('frame', frame)

    # when everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


# inspired by
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html
# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
# https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html