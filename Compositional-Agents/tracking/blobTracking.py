import cv2
import numpy as np


# Set our filtering parameters
# Initialize parameter settiings
params = cv2.SimpleBlobDetector_Params()

# Set thresholds
params.minThreshold = 20
params.maxThreshold = 2000

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.2

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.8

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

# video capture object
cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")

# detector
detector = cv2.SimpleBlobDetector_create(params)


while True:
    ret, frame = cap.read()

    # Detect the blobs in the image
    keypoints = detector.detect(frame)
    print(len(keypoints))

    # Draw detected keypoints as red circles
    fKeyPoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # Display found keypoints
    cv2.imshow("Keypoints", fKeyPoints)

cv2.destroyAllWindows()
