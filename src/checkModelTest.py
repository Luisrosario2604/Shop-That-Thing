#!/usr/bin/env python3.7

import numpy as np
import pickle
import sys
import cv2 as cv

capture = cv.VideoCapture(-1)

def getTheNumber():
    _, frame = capture.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    sensivity = 30
    lower_range = np.array([60 - sensivity, 100, 100])
    upper_range = np.array([60 + sensivity, 255, 255])

    mask_black_white = cv.inRange(hsv, lower_range, upper_range)
    cv.imshow('Normal', frame)
    cv.imshow('Black&White', mask_black_white)

    #hwnd = winGuiAuto.findTopWindow("Black&White")

    return


def checkArguments():
    global file
    if len(sys.argv) >= 2:
        return sys.argv[1]
    else:
        sys.exit(84)


def main():
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        getTheNumber()
    capture.release()
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
