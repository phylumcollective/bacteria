import numpy as np
import cv2
import argparse
import random


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


# threshold = 142 (low threshold, entagled)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    # parser.add_argument("-f", "--filepath", required=True, help="Path to the image file")
    # parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()

# load image, convert to gray and scale down
img = loadImg('../img/layer/DSC_3643.JPG', gray=True)
img = scaleImg(img)
img2 = loadImg('../img/layer/DSC_3643.JPG')
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

approxs = []
gen_seeds = []

# draw the contours (-2 removes plate edges)
for i in range(len(contours)-2):
    # Calculate area and remove small elements
    area = cv2.contourArea(contours[i])
    # -1 in 4th column means it's an external contour
    if hierarchy[0][i][3] == -1 and area > 550 and area < 200000:
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of centroid & draw it
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cY < img.shape[0] - 120 and cX < img.shape[1] - 90 and cY > 10 and cX > 249:
            # wander (randomly perturb contour points in layer growth agency)
            r = random.random()
            rr = random.uniform(-0.004, 0.004)
            rrr = 0
            if r > 0.5:
                rrr = rr
            else:
                rrr = 0
            # contour approximation ("smoothing")
            epsilon = (0.005+rrr)*cv2.arcLength(contours[i], True)  # lots of points (entangled)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            approxs.append(approx)
            # print(approx)
            cv2.drawContours(out1, [approx], -1, (204, 204, 204), 3)
            # approx2 = cv2.approxPolyDP(contours[i], epsilon*0.01, True)
            cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            cv2.drawContours(img2, [approx], -1, (204, 204, 204), 3)
            # cv2.drawContours(out, contours, i, (204, 204, 204), 3)
            # print(contours[i][0][0])
            # for a in approx:
            #     for aa in a:
            #         print(aa)
print("number of contours: " + str(len(approxs)))
for a in approxs:
    for cnt in a:
        coord = cnt[0]
        # remember numpy arrays are row/col while opencv are col/row (as is common for images)
        # print(img_seeds[coord[1]][coord[0]])
        gen_seeds.append(img_seeds[coord[1]][coord[0]])
        # print(coord)

print("number of seeds: " + str(len(gen_seeds)))


cv2.imshow("Image", scaleImg(loadImg('../img/layer/DSC_3643.JPG')))
cv2.imshow("Contours (mask:out)", out)
cv2.imshow("Contours (mask:out1)", out1)
cv2.imshow("Contours", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 115:
        cv2.imwrite('../img/layer/DSC_3643_contours_out.jpg', out)
        cv2.imwrite('../img/layer/DSC_3643_contours_out1.jpg', out1)
        break
    if key == 27:
        break

cv2.destroyAllWindows()
