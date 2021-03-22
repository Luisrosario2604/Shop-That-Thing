#!/usr/bin/env python3.7

import numpy as np
import cv2 as cv
import pickle
import sys

capture = None


#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img / 255
    return img


def checkArgumentsForCamera():
    global capture

    width = 1920
    height = 1080

    if len(sys.argv) == 2:
        capture = cv.VideoCapture(sys.argv[1])
    else:
        capture = cv.VideoCapture(-1)

    capture.set(3, width)
    capture.set(4, height)


def main():
    threshold = 0.65  # MINIMUM PROBABILITY TO CLASSIFY

    checkArgumentsForCamera()

    #### LOAD THE TRAINNED MODEL
    pickle_in = open("model_trained_10.p", "rb")
    model = pickle.load(pickle_in)

    while True:
        success, imgOriginal = capture.read()
        img = np.asarray(imgOriginal)
        img = cv.resize(img, (32, 32))
        img = preProcessing(img)
        cv.imshow("Processsed Image", img)
        img = img.reshape(1, 32, 32, 1)
        #### PREDICT
        classIndex = int(model.predict_classes(img))
        # print(classIndex)
        predictions = model.predict(img)
        # print(predictions)
        probVal = np.amax(predictions)
        print(classIndex, probVal)

        if probVal > threshold:
            cv.putText(imgOriginal, str(classIndex) + "   " + str(probVal),
                        (50, 50), cv.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255), 1)

        cv.imshow("Original Image", imgOriginal)
        if cv.waitKey(1) and 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # execute only if run as a script
    main()
