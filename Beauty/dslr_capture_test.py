import sys
import os
import time
import logging
import argparse
import cv2
import numpy as np
sys.path.insert(0, '../utils/')  # adding utils folder to the system path
import gphoto2 as gp


def loadImg(s, read_as_float32=False, gray=False):
    if read_as_float32:
        img = cv2.imread(s).astype(np.float32) / 255
    else:
        img = cv2.imread(s)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def setupCamera(target="Memory card"):
    # open camera connection
    camera = gp.Camera()
    camera.init()

    # get configuration tree
    config = camera.get_config()

    # find the capture target config item
    capture_target = config.get_child_by_name('capturetarget')

    # set value to Memory card (default) or Internal RAM
    # value = capture_target.get_value()
    capture_target.set_value(target)
    # set config
    camera.set_config(config)

    return camera


def main():
    prevTime = 0  # will store last time capture time updated
    INTERVAL = args.interval  # 15 minutes = 900 seconds

    DIMENSIONS = (600, 400)

    # use Python logging
    logging.basicConfig(
        format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    callback_obj = gp.check_result(gp.use_python_logging())

    # open camera connection
    camera = setupCamera()

    # capture image from camera
    try:
        while True:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            currTime = time.time()
            if(currTime-prevTime >= INTERVAL):
                prevTime = currTime
                print('Capturing image')
                file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
                print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
                # rename the file with a timestamp
                if file_path.name.lower().endswith(".jpg"):
                    new_filename = "{}.jpg".format(timestr)
                target = os.path.join('./captures', new_filename)
                print('Copying image to', target)
                camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                camera_file.save(target)
                # load image, draw frame using opengl and send it to syphon
                img = loadImg(target)
                img = cv2.resize(img, DIMENSIONS)
                imgcvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv uses bgr so we have to convert
                cv2.imshow('Camera image', img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()
    finally:
        # clean up
        camera.exit()
        print("done")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", type=int, required=False, default=1, help="timelapse interval (default=1 second)")
    args = parser.parse_args()
    sys.exit(main())
