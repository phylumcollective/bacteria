# based on code by Sergio Canu:
# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/

import cv2
import numpy as np
import distanceTracker


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# Create tracker object
tracker = distanceTracker.EuclideanDistTracker()

# corner tracking parms and lk params for optical flow
# cornerTrackParams = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
lkParams = dict(winSize=(20, 20), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# set up video capture
cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")

# read the frames, rescale and convert to grayscale
ret, prevFrame = cap.read()
prevFrame = rescaleFrame(prevFrame)
prevFrameGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
prevFrameGray = cv2.medianBlur(prevFrameGray, 7)

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

# modify the data type
# setting to 32-bit floating point
# initializing output images
averageVal = np.float32(prevFrameGray)

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
   history=100, varThreshold=25)

# kernel for dilation
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    # height, width, _ = frame.shape
    # frame = cv2.resize(frame, (667, 500), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    # convert to grayscale & resize
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = rescaleFrame(frameGray)
    frameGray = cv2.medianBlur(frameGray, 7)
    # Extract Region of interest
    # roi = frame[340: 720, 500: 800]
    roi = frame.copy()
    roi = rescaleFrame(roi)

    # 1. Object Detection
    # === using bg subtraction (for moving objects) ===
    mask = object_detector.apply(frameGray)
    ret, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # dilate to fill in gaps
    mask = cv2.dilate(mask, None, kernel, iterations=3)

    # === compute a running average to smooth out jitter/flicker
    # https://cvexplained.wordpress.com/2020/04/17/running-average-model-background-subtraction/
    # using the cv2.accumulateWeighted() function that updates the running average
    # cv2.accumulateWeighted(srcImage, outputImage, alphaValue)
    # higher alpha value = it updates faster...
    cv2.accumulateWeighted(mask, averageVal, 0.75)

    # convert the matrix elements to absolute values and converting the result back to 8-bit.
    runningAvg = cv2.convertScaleAbs(averageVal)

    # find contours
    contours, hierarchy = cv2.findContours(runningAvg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for i in range(len(contours)):
        # Calculate area and remove small elements
        area = cv2.contourArea(contours[i])
        # -1 in 4th column means it's an external contour
        if hierarchy[0][i][3] == -1 and area > 100:
            cv2.drawContours(roi, contours, i, (200, 194, 200), 1)
            x, y, w, h = cv2.boundingRect(contours[i])
            detections.append([x, y, w, h, contours[i], 0])

    # detections = []
    # for cnt in contours:
    #     # Calculate area and remove small elements
    #     area = cv2.contourArea(cnt)
    #     if area > 100:
    #         cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    #         x, y, w, h = cv2.boundingRect(cnt)
    #
    #         detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id, cntr, hull = box_id
        cv2.putText(roi, str(id), (x, y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)
        # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 3. Optical flow
    # calculate LK opical flow and get good points for the pixels that are moving
    prevPts = prevPoints
    if contours:  # make sure list is not empty
        for i in range(len(contours)):
            # -1 in 4th column means it's an external contour
            if hierarchy[0][i][3] == -1 and area > 100:
                # copy interest points
                prevPts = np.array(contours[i], dtype=np.float32)

                vfield = np.zeros_like(prevFrame)

                # contoursArr = np.array(contours[0], dtype=np.float32)
                nextPts, status, err = cv2.calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPts, None, **lkParams)

                # get good points to draw
                if nextPts is not None:
                    goodNew = nextPts[status == 1]  # status vector set to 1 if flow has been found
                    goodPrev = prevPts[status == 1]

                    # calculate the x/y positions of the points
                    for i, (new, prev) in enumerate(zip(goodNew, goodPrev)):
                        xNew, yNew = new.ravel()
                        xPrev, yPrev = prev.ravel()

                        # draw the points
                        cv2.line(vfield, (int(xNew), int(yNew)), (int(xPrev), int(yPrev)), (127, 125, 125), 1)
                        # cv2.rectangle(vfield, (int(xNew), int(yNew)), (int(xNew+5), int(yNew-5)), (125, 125, 125), 1)

                    roi = cv2.add(roi, vfield)

    # update previous frame
    prevFrameGray = frameGray.copy()

    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", rescaleFrame(frame))
    cv2.imshow("Mask", runningAvg)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
