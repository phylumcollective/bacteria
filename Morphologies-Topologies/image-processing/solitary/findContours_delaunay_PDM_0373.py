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


# draw Delaunay triangles
def drawDelaunay(img, subdiv, delaunayColor):
    triangleList = subdiv.getTriangleList()
    triangleList = np.array(triangleList, dtype=np.int32)
    size = img.shape
    r = (0, 0, size[1], size[0])

    triangleListNew = []

    for t in triangleList:
        triangleListNew.append([t[0], t[1], t[2], t[3], t[4], t[5]])

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3):
            cv2.line(img, pt1, pt2, delaunayColor, 3)
            cv2.line(img, pt2, pt3, delaunayColor, 3)
            cv2.line(img, pt3, pt1, delaunayColor, 3)

    triangleListNew = np.array(triangleListNew, dtype=np.int32)
    return triangleListNew


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# thresh 90
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/solitary/PDM_0373.JPG', gray=True)
img = scaleImg(img)
img2 = loadImg('../img/solitary/PDM_0373.JPG')
img2 = scaleImg(img2)

# create the random seeds based upon image dimensions
# so we have an x/y grid of seeds (that correspond to pixel coordinates) (x*y+1)
img_seeds = np.arange(1, (img.shape[0]*img.shape[1]) + 1).reshape(img.shape)

# blur & threshold
imgBlur = cv2.medianBlur(img, 13)  # limit blur (manifold)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(thresh)
out1 = np.zeros_like(thresh)

# Rectangle to be used with Subdiv2D
rect = (0, 0, img.shape[1], img.shape[0])
# Create an instance of Subdiv2D
subdiv = cv2.Subdiv2D(rect)

delaunayPts = []
gen_seeds = []

# draw the contours (-2 removes plate edges)
for i in range(len(contours)-2):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 190 and area < 3000000:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 550 and cX < img.shape[1] - 300 and (cY * cX) > 99000 and cY > 300 and cX > 200:
            # contour approximation ("smoothing")
            epsilon = 0.05*cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            subdiv.insert(approx)
            # print(approx)
            # approx2 = cv2.approxPolyDP(contours[i], epsilon*0.01, True)
            cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            cv2.drawContours(out1, contours, i, (204, 204, 204), 3)
            cv2.drawContours(img2, contours, i, (204, 204, 204), 3)

        # draw Delaunay triangles
        pts = drawDelaunay(out1, subdiv, (204, 204, 204))
        drawDelaunay(img2, subdiv, (204, 204, 204))
        delaunayPts.append(pts)

triCount = 0
for t in delaunayPts:
    for tri in t:
        # print(tri)
        triCount += 1
        # remember numpy arrays are row/col while opencv are col/row (as is common for images)
        gen_seeds.append(img_seeds[tri[1]][tri[0]])
        gen_seeds.append(img_seeds[tri[3]][tri[2]])
        gen_seeds.append(img_seeds[tri[5]][tri[4]])

print("number of triangles: " + str(triCount))

cv2.imshow("Image", scaleImg(loadImg('../img/solitary/PDM_0373.JPG')))
cv2.imshow("Contours (mask:out)", out)
cv2.imshow("Contours (mask:out1)", out1)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 115:
        cv2.imwrite('../img/solitary/PDM_0373_contours_out.jpg', out)
        cv2.imwrite('../img/solitary/PDM_0373_contours_out1.jpg', out1)
        break
    if key == 27:
        break

cv2.destroyAllWindows()
