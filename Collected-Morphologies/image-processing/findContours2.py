import cv2
import numpy as np


def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 3)
    img_canny = cv2.Canny(img_gray, 0, 41)
    kernel = np.ones((7, 7))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=3)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


def draw_contours(img):
    contours, hierarchy = cv2.findContours(process(img), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # cnt = max(contours, key=cv2.contourArea)
    # peri = cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, 0.004 * peri, True)
    # cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)
    # draw the contours
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(img, contours, i, (0, 255, 0), 2)


img = cv2.imread("img/Beauty_PA_2021_08_05_15_53_20_0002.JPG")
h, w, c = img.shape

img = cv2.resize(img, (w // 2, h // 2))
draw_contours(img)

cv2.imshow("Image", img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
