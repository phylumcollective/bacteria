"""
Object tracking. Find contours in a video and send either
the boundingRect coordinates or the convex hull points via OSC
"""

import cv2
import argparse
import time
import numpy as np
from pythonosc import udp_client, osc_message_builder, osc_bundle_builder
import sys


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def sendContour(id, area, x=0, y=0, w=0, h=0, hull=None, addr="/swarm/contour"):
    # send OSC Message representing a contour bounding box in the form of:
    # ["/swarm/contour", id, area, x, y, w, h, hull, addr]
    msg = osc_message_builder.OscMessageBuilder(address=addr)
    msg.add_arg(id, arg_type="i")
    msg.add_arg(area, arg_type="f")
    # if convex hull point are passed, send those (instead of bounding box pts)
    if hull is not None:
        msg.add_arg(
            str(hull), arg_type="s"
        )  # send as a string since OSC doesn't like arrays
    else:
        msg.add_arg(x, arg_type="i")
        msg.add_arg(y, arg_type="i")
        msg.add_arg(w, arg_type="i")
        msg.add_arg(h, arg_type="i")
    oscClient.send(msg.build())
    # arr = np.array(pts, dtype='int32')
    # # arr1 = np.append(arr, [[[-1, -1]]], axis=0)  # mark end of the array
    # msg.add_arg(arr.tobytes(), arg_type='b')  # convert to bytes
    # oscClient.send(msg.build())
    # print(str(id) + ": " + str(x) + "," + str(y) + "," + str(w) + "," + str(h))


def sendContours(boxIds, addr="/swarm/contour"):
    # send OSC Bundle with OSC messages representing all contour bounding boxes:
    # ["/swarm/contour", id, area, x, y, w, h]
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    for boxId in boxIds:
        x, y, w, h, id, cntr, hull = boxId
        msg = osc_message_builder.OscMessageBuilder(address=addr)
        msg.add_arg(id, arg_type="i")
        msg.add_arg(cv2.contourArea(cntr), arg_type="f")
        if args.sendHull:
            msg.add_arg((str(hull)), arg_type="s")
        else:
            msg.add_arg(x, arg_type="i")
            msg.add_arg(y, arg_type="i")
            msg.add_arg(w, arg_type="i")
            msg.add_arg(h, arg_type="i")
        bundle.add_content(msg.build())
        # arr = np.array(pts, dtype='int32')
        # # arr1 = np.append(arr, [[[-1, -1]]], axis=0)  # mark end of the array
        # msg.add_arg(arr.tobytes(), arg_type='b')  # convert to bytes
        # bundle.add_content(msg.build())

    oscClient.send(bundle.build())


if __name__ == "__main__":
    print(__doc__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument(
        "--port", type=int, default=5005, help="The port the OSC server is listening on"
    )
    parser.add_argument(
        "--bundle",
        type=bool,
        default=False,
        help="Send contours as an OSC Bundle (instead of individual OSC Messages)",
    )
    parser.add_argument(
        "--drawHull",
        type=bool,
        default=False,
        help="Draw a convexHull version of the points instead of the full contours",
    )
    parser.add_argument(
        "--sendHull",
        type=bool,
        default=False,
        help="Send convexHull points instead of the boundingRect points of the contours",
    )
    args = parser.parse_args()

    # adding utils folder to the system path
    sys.path.insert(0, "/Users/carlos/Documents/GitHub/bacteria/utils/")
    import distanceTracker

    # def adjustContrast(image, clip_hist_percent=1):
    #     # Automatic brightness and contrast optimization with optional histogram clipping
    #     # taken from: https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
    #
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     # Calculate grayscale histogram
    #     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    #     hist_size = len(hist)
    #
    #     # Calculate cumulative distribution from the histogram
    #     accumulator = []
    #     accumulator.append(float(hist[0]))
    #     for index in range(1, hist_size):
    #         accumulator.append(accumulator[index - 1] + float(hist[index]))
    #
    #     # Locate points to clip
    #     maximum = accumulator[-1]
    #     clip_hist_percent *= (maximum/100.0)
    #     clip_hist_percent /= 2.0
    #
    #     # Locate left cut
    #     minimum_gray = 0
    #     while accumulator[minimum_gray] < clip_hist_percent:
    #         minimum_gray += 1
    #
    #     # Locate right cut
    #     maximum_gray = hist_size - 1
    #     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
    #         maximum_gray -= 1
    #
    #     # Calculate alpha and beta values
    #     alpha = 255 / (maximum_gray - minimum_gray)
    #     beta = -minimum_gray * alpha
    #
    #     '''
    #     # Calculate new histogram with desired range and show histogram
    #     new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    #     plt.plot(hist)
    #     plt.plot(new_hist)
    #     plt.xlim([0,256])
    #     plt.show()
    #     '''
    #
    #     auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #     return (auto_result, alpha, beta)

    # Create tracker object
    tracker = distanceTracker.EuclideanDistTracker()

    # set up video capture
    cap = cv2.VideoCapture("videos/PA_03-15-21.mp4")
    # cap = cv2.VideoCapture(2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

    # read a frame then convert it to grayscale and rescale
    ret, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = rescaleFrame(frameGray)

    # modify the data type
    # setting to 32-bit floating point
    # initializing output images
    averageVal = np.float32(frameGray)

    # Object detection object
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)

    # kernel for noise removal and dilation
    kernel = np.ones((3, 3), np.uint8)

    # OSC
    oscClient = udp_client.UDPClient(args.ip, args.port)

    while True:
        ret, frame = cap.read()
        # height, width, _ = frame.shape
        # frame = cv2.resize(frame, (667, 500), fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # convert to grayscale & resize
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameGray = rescaleFrame(frameGray)
        # frameGray, alpha, beta = adjustContrast(frame, 0.1)
        frameGray = cv2.medianBlur(frameGray, 7)
        # Extract Region of interest
        # roi = frame[340: 720, 500: 800]
        roi = frame.copy()
        roi = rescaleFrame(roi)

        # 1. Object Detection
        # # === using threshold and distance transform (for static objects) ===
        # ret, frameBlurThresh = cv2.threshold(frameBlur, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # # noise removal
        # opening = cv2.morphologyEx(frameBlurThresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # # get the bg
        # sureBg = cv2.dilate(opening, kernel, iterations=3)
        # distTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # # get the fg
        # ret, sureFg = cv2.threshold(distTransform, 0.8*distTransform.max(), 255, 0)
        # sureFg = np.uint8(sureFg)
        # unknown = cv2.subtract(sureBg, sureFg)
        # ret, markers = cv2.connectedComponents(sureFg)
        # markers = markers+1
        # markers[unknown==255] = 0
        # # apply watershed to segment image
        # markers = cv2.watershed(roi, markers)
        #
        # # find contours
        # contours, hierarchy = cv2.findContours(frameBlurThresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # for i in range(len(contours)):
        #     # -1 in 4th column means it's an external contour
        #     if hierarchy[0][i][3] == -1:
        #         cv2.drawContours(roi, contours, i, (0, 255, 0), 2)

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
        contours, hierarchy = cv2.findContours(
            runningAvg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        # print(contours)

        # portions below based on code by Sergio Canu:
        # https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
        detections = []
        # hulls = []
        for i in range(len(contours)):
            # Calculate area and remove small elements
            area = cv2.contourArea(contours[i])
            hull = cv2.convexHull(contours[i])
            # hulls.append(hull)
            # print("hull: ")
            # sarr = [str(a[0]) for a in hull]
            # print(', '.join(sarr))
            # -1 in 4th column means it's an external contour
            if hierarchy[0][i][3] == -1 and area > 100:
                if args.drawHull:
                    cv2.drawContours(roi, [hull], 0, (200, 194, 200), 1)
                else:
                    cv2.drawContours(roi, contours, i, (200, 194, 200), 1)
                x, y, w, h = cv2.boundingRect(contours[i])
                detections.append([x, y, w, h, contours[i], hull])
                # print(str(i) + ": " + str(contours[i]))

        # detections = []
        # for cnt in contours:
        #     # Calculate area and remove small elements
        #     area = cv2.contourArea(cnt)
        #     if area > 100:
        #         cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        #         x, y, w, h = cv2.boundingRect(cnt)
        #
        #         detections.append([x, y, w, h])

        # print("hulls: ")
        # sHulls = [str(a) for a in hulls]
        # print(','.join(sHulls))
        # convert hulls array to a string
        # sHulls = ','.join([str(a) for a in hulls])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        if detections and args.bundle:
            sendContours(boxes_ids)
        for box_id in boxes_ids:
            x, y, w, h, id, cntr, hull = box_id
            cv2.putText(
                roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1
            )
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 1)
            # send the contour via OSC
            if not args.bundle:
                if args.sendHull:
                    print("hull: " + str(hull))
                    sendContour(id, cv2.contourArea(cntr), hull=hull)
                else:
                    sendContour(id, cv2.contourArea(cntr), x, y, w, h)
                time.sleep(0.001)

        cv2.imshow("ROI", roi)
        cv2.imshow("Frame", rescaleFrame(frame))
        cv2.imshow("Mask", runningAvg)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
