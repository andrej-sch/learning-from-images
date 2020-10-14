# 1. read each frame from the camera (if necessary resize the image)
#    and extract the SIFT features using OpenCV methods
#    Note: use the gray image - so you need to convert the image
# 2. draw the keypoints using cv2.drawKeypoints
#    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

# close the window and application by pressing a key

import cv2 as cv

cap = cv.VideoCapture(0)
cv.namedWindow('Learning from images: SIFT feature visualization', cv.WINDOW_NORMAL)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(frame, None)

    flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    frame = cv.drawKeypoints(frame, kp, frame, flags=flag)

    cv.imshow('Learning from images: SIFT feature visualization', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv.destroyAllWindows()


# inspired by
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html