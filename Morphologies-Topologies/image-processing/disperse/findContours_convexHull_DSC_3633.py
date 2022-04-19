import numpy as np
import cv2
import argparse
from math import sqrt


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


# find the two contour points with longest distance between them (path attribute: balance)
def findLongestDist(cnt):
    # get contour extremes
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    pts = []
    pts.append(leftmost)
    pts.append(rightmost)
    pts.append(topmost)
    pts.append(bottommost)

    n = len(pts)
    maxDist = 0
    maxDistPts = []

    # Iterate over all possible point pairs
    for i in range(n):
        for j in range(i + 1, n):
            # get the current distance and update the max distance
            distance = sqrt(((pts[i][0]-pts[j][0])**2)+((pts[i][1]-pts[j][1])**2))
            if (distance > maxDist):
                maxDistPts = [pts[i], pts[j]]
                maxDist = distance

    # print actual max distance
    # print(maxDist)
    # return points with the longest diatance
    return maxDistPts


# thresh 144
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/disperse/DSC_3633.JPG', gray=True)
img = scaleImg(img)
img2 = loadImg('../img/disperse/DSC_3633.JPG')
img2 = scaleImg(img2)

# create the random seeds based upon image dimensions
# so we have an x/y grid of seeds (that correspond to pixel coordinates) (x*y+1)
img_seeds = np.arange(1, (img.shape[0]*img.shape[1]) + 1).reshape(img.shape)

# blur & threshold
imgBlur = cv2.medianBlur(img, 13)  # limit blur (manifold)

# brightness & contrast adjustment
# https://towardsdatascience.com/contrast-enhancement-of-grayscale-images-using-morphological-operators-de6d483545a1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
# Top Hat Transform
topHat = cv2.morphologyEx(imgBlur, cv2.MORPH_TOPHAT, kernel)
# Black Hat Transform
blackHat = cv2.morphologyEx(imgBlur, cv2.MORPH_BLACKHAT, kernel)
# adjusted
adjusted = imgBlur + topHat - blackHat

ret, thresh = cv2.threshold(adjusted, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(thresh)
out1 = np.zeros_like(thresh)

hull = []
gen_seeds = []
longestPts = []

# draw the contours (-2 removes plate edges)
for i in range(len(contours)-2):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 1:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 1 and cX < img.shape[1] - 570 and (cY * cX) > 99000 and cY > 1 and cX > 460:
            if cX < img.shape[1] - 750:
                # create a convex hull object for each contour
                h = cv2.convexHull(contours[i], False)
                # append to hull list
                hull.append(h)
                hullArea = cv2.contourArea(h)
                longest = findLongestDist(h)  # get the two contour points with the longest distance
                longestPts.append(longest)
                ll = np.array(longest)
                if hullArea > 1:
                    cv2.drawContours(out, contours, i, (204, 204, 204), 3)
                    cv2.drawContours(out1, contours, i, (204, 204, 204), 3)
                    cv2.drawContours(out1, [h], -1, (204, 204, 204), 3)
                    cv2.drawContours(out1, [ll], -1, (204, 204, 204), 3)
                    cv2.drawContours(img2, contours, i, (204, 204, 204), 3)
                    cv2.drawContours(img2, [h], -1, (204, 204, 204), 3)
                    # print("number of hull points: " + str(len(h)))
                    # for hh in h:
                    #     coord = hh[0]
                    #     circle = cv2.circle(img2, coord, 10, (255, 0, 0), 1)


print("number of contours: " + str(len(longestPts)))
for lp in longestPts:
    pt1 = lp[0]
    pt2 = lp[1]

    # remember numpy arrays are row/col while opencv are col/row (as is common for images)
    # print(img_seeds[coord[1]][coord[0]])
    gen_seeds.append(img_seeds[pt1[1]][pt1[0]])
    gen_seeds.append(img_seeds[pt2[1]][pt2[0]])
    # print(coord)

print("number of seeds: " + str(len(gen_seeds)))


cv2.imshow("Image", scaleImg(loadImg('../img/disperse/DSC_3633.JPG')))
cv2.imshow("Contours (mask:out)", out)
cv2.imshow("Contours (mask:out1)", out1)
cv2.imshow("Contours", img2)
cv2.imshow("Adjusted", adjusted)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 115:
        cv2.imwrite('../img/disperse/DSC_3633_contours_out.jpg', out)
        cv2.imwrite('../img/disperse/DSC_3633_contours_out1.jpg', out1)
        break
    if key == 27:
        break

cv2.destroyAllWindows()
