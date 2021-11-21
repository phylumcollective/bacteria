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


def processImg(img):
    # img_blur = cv2.GaussianBlur(img, (3, 3), 3)
    img_canny = cv2.Canny(img, 55, 295)
    kernel = np.ones((9, 9))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/blossom/BMC_0681.JPG', gray=True)
img = scaleImg(img)
img = processImg(img)
img2 = loadImg('../img/blossom/BMC_0681.JPG')
img2 = scaleImg(img2)

# create the random seeds based upon image dimensions
img_seeds = np.arange(1, (img.shape[0]*img.shape[1]) + 1).reshape(img.shape)

# blur & threshold
imgBlur = cv2.medianBlur(img, 1)
ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

# find Contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

out = np.zeros_like(thresh)

approxs = []
gen_seeds = []

# draw the contours
for i in range(len(contours)):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 820:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it (also add the coords to the centerPoints array)
        cX = int(M["m10"] / M["m00"]) # cY > 115 and cX > 160:
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 350 and cX < img.shape[0] - 250 and (cY * cX) > 99000 and cY > 245 and cX > 160:
            # contour approximation ("smoothing")
            epsilon = 0.01*cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            approxs.append(approx)
            # print(approx)
            cv2.drawContours(out, [approx], -1, (204, 204, 204), 3)
            cv2.drawContours(img2, [approx], -1, (204, 204, 204), 3)
            # cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            # print(contours[i][0][0])
            # for a in approx:
            #     for aa in a:
            #         print(aa)
for a in approxs:
    for cnt in a:
        coord = cnt[0]
        # remember numpy arrays are row/col while opencv are col/row (as is common for images)
        print(img_seeds[coord[1]][coord[0]])
        gen_seeds.append(img_seeds[coord[1]][coord[0]])
        print(coord)


cv2.imshow("Image", scaleImg(loadImg('../img/blossom/BMC_0681.JPG')))
cv2.imshow("Contours (mask)", out)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()