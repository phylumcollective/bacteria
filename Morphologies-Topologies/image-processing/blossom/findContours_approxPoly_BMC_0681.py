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


def processImg(img, min, max):
    #img1 = cv2.GaussianBlur(img, (3, 3), 3)
    img_canny = cv2.Canny(img, min, max)
    #img_lap = cv2.Laplacian(img1, cv2.CV_8UC1)
    kernel = np.ones((7, 7), np.uint8)
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


# find the two contour points with longest distance between them
def findLongestDist(cnt):
    # get contour extemes
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/blossom/BMC_0681.JPG', gray=True)
img = scaleImg(img)
img = processImg(img, 55, 260)
img2 = loadImg('../img/blossom/BMC_0681.JPG')
img2 = scaleImg(img2)

# create the random seeds based upon image dimensions
# so we have an x/y grid of seeds (that correspond to pixel coordinates) (x*y+1)
img_seeds = np.arange(1, (img.shape[0]*img.shape[1]) + 1).reshape(img.shape)

# blur & threshold
imgBlur = cv2.medianBlur(img, 1)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(img)

# centroids = []
longestPts = []
gen_seeds = []

# draw the contours
for i in range(len(contours)):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 4:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 350 and cX < img.shape[0] - 155 and (cY * cX) > 99000 and cY > 245 and cX > 10:
            # contour approximation ("smoothing")
            epsilon = 0.0001*cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            # N = cv2.moments(approx)
            # centroidX = int(N["m10"] / N["m00"])
            # centroidY = int(N["m01"] / N["m00"])
            # centroid = (centroidX, centroidY)
            # centroids.append(centroid)
            longest = findLongestDist(contours[i])  # get the two contour points with the longest distance
            longestPts.append(longest)
            # print("the contour:" + str(contours[i]))
            # print("longest distance:" + str(longest))
            # print(approx)
            cv2.drawContours(out, [approx], -1, (204, 204, 204), 3)
            cv2.drawContours(img2, [approx], -1, (204, 204, 204), 3)
            # cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            # print(contours[i][0][0])
            # for a in approx:
            #     for aa in a:
            #         print(aa)
print("number of contours: " + str(len(longestPts)))
for lp in longestPts:
    pt1 = lp[0]
    pt2 = lp[1]

    # remember numpy arrays are row/col while opencv are col/row (as is common for images)
    # print(img_seeds[coord[1]][coord[0]])
    gen_seeds.append(img_seeds[pt1[1]][pt1[0]])
    gen_seeds.append(img_seeds[pt2[1]][pt2[0]])
    # print(coord)

# print("number of contour centroids: " + str(len(centroids)))
# for coord in centroids:
#     # remember numpy arrays are row/col while opencv are col/row (as is common for images)
#     # print(img_seeds[coord[1]][coord[0]])
#     gen_seeds.append(img_seeds[coord[1]][coord[0]])
#     # print(coord)

print("number of seeds: " + str(len(gen_seeds)))

cv2.imshow("Image", processImg(scaleImg(loadImg('../img/blossom/BMC_0681.JPG')), 55, 260))
cv2.imshow("Contours (mask)", out)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 115:
        cv2.imwrite('../img/blossom/BMC_0681_contours.jpg', out)
        break
    if key == 27:
        break

cv2.destroyAllWindows()
