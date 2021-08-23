import cv2
import numpy as np


# Dense Optical flow tracking using the Robust Local Optical Flow (RLOF) method


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# set up video capture
cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)

# read the frames, rescale and convert to grayscale
ret, prevFrame = cap.read()
prevFrame = rescaleFrame(prevFrame)
# prevFrameBlur = cv2.medianBlur(prevFrameGray, 25)

# create HSV & make Value a constant
hsv = np.zeros_like(prevFrame)
hsv[..., 1] = 255

while True:
    ret, frame = cap.read()
    frame = rescaleFrame(frame)
    frameCopy = frame
    if not ret:
        break

    # RLOF dense Optical flow
    flow = cv2.optflow.calcOpticalFlowDenseRLOF(prevFrame, frame, None)

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Use Hue and Saturation to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', frameCopy)
    cv2.imshow("optical flow", bgr)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # update reference
    prevFrame = frame


cap.release()
cv2.destroyAllWindows()
