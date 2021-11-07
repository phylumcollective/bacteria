import numpy as np
import cv2, argparse


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
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    triangleList = np.array(triangleList, dtype=np.int32)
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 2)
            cv2.line(img, pt2, pt3, delaunay_color, 2)
            cv2.line(img, pt3, pt1, delaunay_color, 2)


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()


# load image, convert to gray and scale down
img = loadImg('img/DSC_3574.JPG', gray=True)
img = scaleImg(img)
img2 = loadImg('img/DSC_3574.JPG')
img2 = scaleImg(img2)

# blur & threshold
imgBlur = cv2.medianBlur(img, 15)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# mask
out = np.zeros_like(thresh)

# Rectangle to be used with Subdiv2D
size = img.shape
rect = (0, 0, size[1], size[0])
# Create an instance of Subdiv2D
subdiv = cv2.Subdiv2D(rect)

# draw the contours (-2 removes plate edges)
for i in range(len(contours)-2):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 566:
        # calculate moments and find centroid
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it (also add the coords to the centerPoints array)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cX < size[1] - 250:
            subdiv.insert((cX, cY))
            # cv2.circle(out, (cX, cY), 3, (255, 255, 255), -1)
            # draw contours
            cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            cv2.drawContours(img2, contours, i, (204, 204, 204), 3)
            # draw contour bounding rectangle
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.putText(out, str(i), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
        # draw Delaunay triangles
        draw_delaunay(out, subdiv, (127, 127, 127))
        draw_delaunay(img2, subdiv, (255, 255, 255))


cv2.imshow("Image", scaleImg(loadImg('img/DSC_3574.JPG')))
cv2.imshow("Contours (mask)", out)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
