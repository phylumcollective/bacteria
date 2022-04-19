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
img = loadImg('img/blossom/DSC_3574.JPG')
img = scaleImg(img)
# img = cv2.equalizeHist(img)
# blur
# imgBlur = cv2.medianBlur(img, 15)

# Change the background to black in order to extract better results during the Distance Transform
src = img.copy()
src[np.all(src < 145, axis=2)] = 0

# Create a kernel that we will use to sharpen our image
# an approximation of second derivative, a quite strong kernel
kernel0 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel0)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian
# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

# Create binary image from source image
bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(bw, 170, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Perform the distance transform algorithm
# dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
dist = cv2.erode(opening, kernel, iterations=3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

# This will be the markers for the foreground objects
ret, sure_fg = cv2.threshold(dist, 0.9, 1.0, cv2.THRESH_BINARY)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# threshold
# ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
# # noise removal
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# # sure background area
# sure_bg = cv2.dilate(opening, kernel, iterations=2)
# # find sure foreground area
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
# apply the watershed algorithm
markers = cv2.watershed(src, markers)

# find contours on the markers
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(thresh)

# for every entry in contours
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1 and area > 25000:
        # We can now draw the external contours from the list of contours
        cv2.drawContours(out, contours, i, (255, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 2)


cv2.imshow("src", out)
cv2.imshow("img", img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
