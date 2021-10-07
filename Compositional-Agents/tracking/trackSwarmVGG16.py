'''
Object tracking using a VGG16 model trained with a custom dataset of our swarming bacteria
'''

import cv2
import numpy as np
import argparse
from tensorflow.keras.applications import vgg16

if __name__ == "__main__":
    print(__doc__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default="127", help="The cutoff for the threshold algorithm (0-255)")
    args = parser.parse_args()

    # the class names
    CLASSES = ["snake", "swirl"]

    # get a different color array for each of the classes
    COLORS = ([255, 0, 0], [0, 255, 0])

    # load the VGG16 model
    model = cv2.dnn.readNetFromTensorflow("../classification/models/vgg16/0000_swarming/frozen_model/frozen_graph.pb")

    # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # set up video capture
    cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")

    # kernel for noise removal and dilation
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]
        w = frame.shape[1]
        image = np.array(frame)

        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.medianBlur(gray, 15)
        ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # dilate to fill in gaps
        thresh = cv2.dilate(thresh, None, kernel, iterations=3)

        # find contours on the markers
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours))

        # for every entry in contours
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # last column in the array is -1 if an external contour (no contours inside of it)
            if hierarchy[0][i][3] == -1 and area > 200:
                # We can now draw the external contours from the list of contours
                # cv2.drawContours(src, contours, i, (0, 255, 0), 2)
                # get the bounding box coordinates & width & height
                x, y, w, h = cv2.boundingRect(contours[i])
                # create blob from image and use that as input to the model
                model.setInput(cv2.dnn.blobFromImage(image[y:y+h, x:x+w].copy(), size=(224, 224), swapRB=False, crop=False))
                # forward propagate the network
                detection = model.forward()
                print(detection)

                # since we are doing binary classification
                # we extract the confidence (i.e., probability) associated with the prediction
                # and use that to determine the class
                class_id = np.where(detection[0][0]  > 0.5, 1,0)
                # map the class id to the class
                class_name = CLASSES[class_id]
                color = COLORS[class_id]
                # color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
                # draw a rectangle around each detected object
                cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
                # put the class name text on the detected object
                cv2.putText(image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if class_id == 0:
                    confidence = str(" (" + str(1 - detection[0]) + ")")
                    cv2.putText(image, confidence, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    confidence = str(" (" + str(detection[0]) + ")")
                    cv2.putText(image, confidence, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('img', image)
        cv2.imshow("src", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
