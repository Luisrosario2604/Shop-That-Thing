#!/usr/bin/env python3.7

import sys
import cv2 as cv

capture = None
file = open('dataSquares.txt', 'a')


def getTheSquare(selecter):

    _, frame = capture.read()

    if selecter:
        r = cv.selectROI(frame)
        if r == (0, 0, 0, 0):
            return 0
        imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        #cv.imshow('Cropped', imCrop)
        print(r)
        return r, frame
    else:
        cv.imshow('Camera', frame)
    return


def saveData(r, frame):
    lastAnswer = input("Validate ? (y or n)\n")
    if lastAnswer == "y":
        lastAnswer = input("Name of the object ?\n")
        if ' ' not in lastAnswer:
            imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            cv.imshow('Cropped', imCrop)
            print(r)
            print("Select price")
            cv.destroyWindow('ROI selector')
            r1 = cv.selectROI(frame)
            print("Select square")
            cv.destroyWindow('ROI selector')
            r2 = cv.selectROI(frame)

            obj = lastAnswer + " " + str(r) + " " + str(r1) + " " + str(r2) + "\n"
            print("Saved !", obj)
            file.write(obj)
            return
    print("Canceled")


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
    global capture

    checkArgumentsForCamera()

    while True:
        if cv.waitKey(33) == ord('q'):
            break
        if cv.waitKey(33) == ord('n'):
            cv.destroyWindow('Camera')
            r, frame = getTheSquare(True)
            cv.imwrite('./green.jpg', frame)
            if r != 0:
                saveData(r, frame)
            cv.destroyWindow('Cropped')
            cv.destroyWindow('ROI selector')
        else:
            getTheSquare(False)

    capture.release()
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
