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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()


# load image, convert to gray and scale down
img = loadImg('img/DSC_3574.JPG', gray=True)
img = scaleImg(img)

# blur & threshold
imgBlur = cv2.medianBlur(img, 15)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

out = np.zeros_like(thresh)

# draw the contours
for i in range(len(contours)):
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(out, contours, i, (204, 204, 204), 3)

# the roi variables
x, y, w, h = args.roi
# select roi
roi = out[int(y):int(y)+int(h), int(x):int(x)+int(w)].copy()

cv2.imshow("Image", scaleImg(loadImg('img/DSC_3574.JPG')))
cv2.imshow("Contours", scaleImg(roi))

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
