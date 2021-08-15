import cv2
import numpy as np

# https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer
# https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# corner tracking parameters for ShiTomasi corner detection
# cornerTrackParams = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=3)
lkParams = dict(winSize=(20, 20), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# set up video capture
cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)

# read the frames, rescale and convert to grayscale
ret, prevFrame = cap.read()
prevFrame = rescaleFrame(prevFrame)
prevFrameGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
# prevFrameBlur = cv2.medianBlur(prevFrameGray, 25)

# Points to track
# prevPts = cv2.goodFeaturesToTrack(prevFrameBlur, mask=None, **cornerTrackParams)
# generate interest points to track
h, w, c = prevFrame.shape
pts = []
for i in range(0, w, 10):
    for j in range(0, h, 10):
        pts.append([[i, j]])
# color = np.random.randint(0,255,(len(pts),3))
prevPoints = np.array(pts, dtype="float32")

# create mask for image drawing
lkmask = np.zeros_like(prevFrame)

# Create random colors
color = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    frame = rescaleFrame(frame)
    # height, width, _ = frame.shape
    # frame = cv2.resize(frame, (667, 500), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    # convert to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frameBlur = cv2.medianBlur(frameGray, 25)

    # copy interest points
    prevPts = prevPoints

    vfield = np.zeros_like(prevFrame)

    # Optical flow
    # calculate LK opical flow and get good points for the pixels that are moving
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPts, None, **lkParams)

    # get good points to draw
    if nextPts is not None:
        goodNew = nextPts[status==1]
        goodPrev = prevPts[status==1]

        # draw the tracks
        for i, (new, prev) in enumerate(zip(goodNew, goodPrev)):
            xNew, yNew = new.ravel()
            xPrev, yPrev = prev.ravel()

            #lkmask = cv2.line(lkmask, (int(xNew), int(yNew)), (int(xPrev), int(yPrev)), color[i].tolist(), 2)
            #frame = cv2.circle(frame, (int(xNew), int(yNew)), 5, color[i].tolist(), -1)
            cv2.line(vfield, (int(xNew), int(yNew)), (int(xPrev), int(yPrev)), (127, 125, 125), 1)
            # cv2.rectangle(vfield, (int(xNew), int(yNew)), (int(xNew+5), int(yNew-5)), (125, 125, 125), 1)

    img = cv2.add(frame, vfield)
    cv2.imshow('LK Optical Flow', img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # update reference
    prevFrameGray = frameGray.copy()
    # prevPts = goodNew.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
