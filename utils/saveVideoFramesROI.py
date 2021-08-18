import cv2, argparse, time


def rescaleFrame(frame, scaleFactor=0.5):
    width = int(frame.shape[1] * scaleFactor)
    height = int(frame.shape[0] * scaleFactor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to source video (mp4) file")
    parser.add_argument("-o", "--output", default="./img_dest/", type=str, help="path to destination for image files")
    parser.add_argument("-r", "--roi", required=True, nargs="+", help="the x/y and width/height of the roi")
    args = parser.parse_args()
    print(args.input)
    print(args.output)
    print(args.roi)

    # set up video capture
    cap = cv2.VideoCapture(args.input)

    count = 0

    # the roi variables
    x, y, w, h = args.roi

    while True:
        ret, frame = cap.read()

        roi = frame[int(y):int(y)+int(h), int(x):int(x)+int(w)].copy()

        # save the new images
        frameCount = f"{count:04d}"
        outfile = '%s%s.jpg' % (args.output, frameCount)
        cv2.imwrite(outfile, roi)
        time.sleep(0.001)

        cv2.imshow("roi", roi)
        count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
