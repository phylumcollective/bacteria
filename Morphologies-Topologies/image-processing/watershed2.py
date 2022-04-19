import numpy as np
import cv2


def loadImg(s, read_as_float32=False, gray=False):
    if read_as_float32:
        img = cv2.imread(s).astype(np.float32) / 255
    else:
        img = cv2.imread(s)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def scaleImg(img, scaleFactor=0.5):
    width = int(img.shape[1] * scaleFactor)
    height = int(img.shape[0] * scaleFactor)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


# load image
img = loadImg('img/layer/DSC_3643.JPG', gray=True)
img = scaleImg(img)

# img = cv2.equalizeHist(img)
# blur
# imgBlur = cv2.medianBlur(img, 15)

# Create binary image from source image
ret, thresh = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Perform the distance transform algorithm
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
# dist = cv2.erode(opening, kernel, iterations=3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
# cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

# This will be the markers for the foreground objects
ret, sure_fg = cv2.threshold(dist, 1, 255, cv2.THRESH_BINARY)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
# apply the watershed algorithm
markers = cv2.watershed(img, markers)

# find contours on the markers
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# for every entry in contours
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        # We can now draw the external contours from the list of contours
        cv2.drawContours(img, contours, i, (0, 255, 0), 2)

cv2.imshow("img", markers)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
