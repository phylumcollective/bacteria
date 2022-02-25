import numpy as np
import cv2
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/disperse/BSM_1799.JPG', gray=True)
img = scaleImg(img)
img2 = loadImg('../img/disperse/BSM_1799.JPG')
img2 = scaleImg(img2)

# create the random seeds based upon image dimensions
# so we have an x/y grid of seeds (that correspond to pixel coordinates) (x*y+1)
img_seeds = np.arange(1, (img.shape[0]*img.shape[1]) + 1).reshape(img.shape)

# blur & threshold
imgBlur = cv2.medianBlur(img, 15)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(thresh)

hull = []
gen_seeds = []

# draw the contours (-2 removes plate edges)
for i in range(len(contours)-2):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 180:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 420 and cX < img.shape[1] - 400 and (cY * cX) > 99000 and cY > 300 and cX > 120:
            # create a convex hull object for each contour
            h = cv2.convexHull(contours[i], False)
            hull.append(h)
            cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            cv2.drawContours(out, [h], -1, (204, 204, 204), 3)
            cv2.drawContours(img2, contours, i, (204, 204, 204), 3)
            cv2.drawContours(img2, [h], -1, (204, 204, 204), 3)


# triCount = 0
# for t in hull:
#     for tri in t:
#         print(tri)
#         triCount += 1
#         # remember numpy arrays are row/col while opencv are col/row (as is common for images)
#         gen_seeds.append(img_seeds[tri[1]][tri[0]])
#         gen_seeds.append(img_seeds[tri[3]][tri[2]])
#         gen_seeds.append(img_seeds[tri[5]][tri[4]])


cv2.imshow("Image", scaleImg(loadImg('../img/disperse/BSM_1799.JPG')))
cv2.imshow("Contours (mask)", out)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 115:
        cv2.imwrite('../img/solitary/DSC_3577_contours.jpg', out)
        break
    if key == 27:
        break

cv2.destroyAllWindows()
