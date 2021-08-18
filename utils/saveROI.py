'''
Create an ROI and save it as an image
'''


import cv2
import argparse
from glob import glob
from datetime import datetime


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    print(__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./img_src/", type=str, help="path to source image (jpg) file")
    parser.add_argument("-o", "--output", default="./img_dest/", type=str, help="path to destination for image files")
    parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()
    print(args.input)
    print(args.output)
    print(args.roi)

    count = 0

    # the roi variables
    x, y, w, h = args.roi
    for fn in glob(args.input + '*.JPG'):
        # load the image
        img = cv2.imread(fn)
        # select roi
        roi = img[int(y):int(y)+int(h), int(x):int(x)+int(w)].copy()

        # save the new images
        frameCount = f"{count:02d}"
        now = datetime.now()
        outfile = '%s%s%s.jpg' % (args.output, frameCount, "_" + now.strftime("%Y-%m-%d_%H%M%S"))
        cv2.imwrite(outfile, roi)
        count += 1

    print('Done')

    cv2.destroyAllWindows()
