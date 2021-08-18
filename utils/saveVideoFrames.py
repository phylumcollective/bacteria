import cv2, argparse, time
import numpy as np
import distanceTracker


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to source video (mp4) file")
    parser.add_argument("-o", "--output", default="./img_save/", type=str, help="path to destination for image files")
    args = parser.parse_args()
    print(args.input)
    print(args.output)

    # Create tracker object
    tracker = distanceTracker.EuclideanDistTracker()

    # set up video capture
    cap = cv2.VideoCapture(args.input)

    # read a frame and convert it to grayscale
    ret, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # modify the data type
    # setting to 32-bit floating point
    # initializing output images
    averageVal = np.float32(frameGray)

    # Object detection object
    object_detector = cv2.createBackgroundSubtractorMOG2(
       history=100, varThreshold=25)

    # kernel for noise removal and dilation
    kernel = np.ones((3, 3), np.uint8)

    count = 0

    while True:
        ret, frame = cap.read()

        # convert to grayscale & resize
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frameGray = rescaleFrame(frameGray)
        # frameGray, alpha, beta = adjustContrast(frame, 0.1)
        frameGray = cv2.medianBlur(frameGray, 11)

        roi = frame.copy()

        # === using bg subtraction (for moving objects) ===
        mask = object_detector.apply(frameGray)
        ret, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # dilate to fill in gaps
        mask = cv2.dilate(mask, None, kernel, iterations=3)

        # === compute a running average to smooth out jitter/flicker
        # https://cvexplained.wordpress.com/2020/04/17/running-average-model-background-subtraction/
        # using the cv2.accumulateWeighted() function that updates the running average
        # cv2.accumulateWeighted(srcImage, outputImage, alphaValue)
        # higher alpha value = it updates faster...
        cv2.accumulateWeighted(mask, averageVal, 0.75)

        # convert the matrix elements to absolute values and converting the result back to 8-bit.
        runningAvg = cv2.convertScaleAbs(averageVal)

        # find contours
        contours, hierarchy = cv2.findContours(runningAvg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # portions below based on code by Sergio Canu:
        # https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
        detections = []

        for i in range(len(contours)):
            # Calculate area and remove small elements
            area = cv2.contourArea(contours[i])
            # -1 in 4th column means it's an external contour
            if hierarchy[0][i][3] == -1 and area > 100:
                # cv2.drawContours(frame, contours, i, (200, 194, 200), 1)
                x, y, w, h = cv2.boundingRect(contours[i])
                detections.append([x, y, w, h, contours[i], 0])
                # print(str(i) + ": " + str(contours[i]))

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id, cntr, hull = box_id
            # cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 1)
            # save the contents within the box as a jpg
            # https://stackoverflow.com/questions/57766374/how-to-save-the-region-of-interest-of-a-frame-using-opencv
            imgSave = frame[y:y+h, x:x+w].copy()
            # https://stackoverflow.com/questions/134934/display-number-with-leading-zeros
            frameCount = f"{count:05d}"
            idCount = f"{id:04d}"
            # https://stackoverflow.com/questions/44535290/save-images-in-loop-with-different-names/44535498
            outfile = '%s%s_%s.jpg' % (args.output, frameCount, idCount)
            cv2.imwrite(outfile, imgSave)
            time.sleep(0.001)

        cv2.imshow("roi", roi)
        count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
