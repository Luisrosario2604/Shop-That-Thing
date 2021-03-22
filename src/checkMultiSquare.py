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


def seeCropped(imgOriginal, squareNumber, splitArray):
    square = splitArray[squareNumber]
    name = square[0] + " Square"
    img = np.asarray(imgOriginal)

    imCrop = img[76:76 + 398, 434:434 + 450]
    cv.imshow('Cropped', imCrop)

    r1 = square[1].replace(" ", "").replace("(", "").replace(")", "").split(',')
    r2 = square[2].replace(" ", "").replace("(", "").replace(")", "").split(',')
    r3 = square[3].replace(" ", "").replace("(", "").replace(")", "").split(',')

    img1 = img[int(r1[1]):int(r1[1]) + int(r1[3]), int(r1[0]):int(r1[0]) + int(r1[2])]
    img2 = img[int(r2[1]):int(r2[1]) + int(r2[3]), int(r2[0]):int(r2[0]) + int(r2[2])]
    img3 = img[int(r3[1]):int(r3[1]) + int(r3[3]), int(r3[0]):int(r3[0]) + int(r3[2])]

    print(r1[0], r1[1], r1[2], r1[3])
    cv.imshow(name + " all", img1)
    cv.imshow(name + " price", img2)
    cv.imshow(name + " square", img3)


def squareFromImage(imgOriginal, squareNumber, splitArray):
    square = splitArray[squareNumber]
    name = square[0] + " Square"
    r = square[3].replace(" ", "").replace("(", "").replace(")", "").split(',')

    img = np.asarray(imgOriginal)
    img = img[int(r[1]):int(r[1]) + int(r[3]), int(r[0]):int(r[0]) + int(r[2])]
    cv.imshow(name + " original", img)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #https://stackoverflow.com/questions/48528754/what-are-recommended-color-spaces-for-detecting-orange-color-in-open-cv
    mask = cv.inRange(hsv, (85, 30, 20), (130, 255, 255))
    #mask = cv.inRange(hsv, (95, 100, 20), (115, 255, 255))
    cv.imshow(name + " colored", mask)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        cv.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv.putText(img, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(approx)
            aspectRatio = float(w) / h
            print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv.putText(img, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv.putText(img, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 5:
            cv.putText(img, "Pentagon", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 10:
            cv.putText(img, "Star", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        #else:
            #cv.putText(img, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv.imshow("shapes", img)


def numberFromImage(imgOriginal, model, squareNumber, splitArray):

    square = splitArray[squareNumber]
    name = square[0] + " Number"
    r = square[2].replace(" ", "").replace("(", "").replace(")", "").split(',')

    img = np.asarray(imgOriginal)
    img = img[int(r[1]):int(r[1]) + int(r[3]), int(r[0]):int(r[0]) + int(r[2])]
    cv.imshow(name + " original", img)
    img = cv.resize(img, (32, 32))
    img = preProcessing(img)
    cv.imshow(name, img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)

    return classIndex, probVal


def mainLoop(splitArray):
    checkArgumentsForCamera()

    pickle_in = open("model_trained_10.p", "rb")
    model = pickle.load(pickle_in)

    while True:
        _, imgOriginal = capture.read()
        cv.imshow("Original Image", imgOriginal)

        #nm, proba = numberFromImage(imgOriginal, model, 0, splitArray)
        #if proba > 0.90:
        #    print(nm, proba)

        squareFromImage(imgOriginal, 0, splitArray)

        #seeCropped(imgOriginal, 0, splitArray)
        if cv.waitKey(33) and 0xFF == ord('q'):
            break


def checkSetUp():
    splitArray = []
    file = open('dataSquares.txt', 'r')
    lines = [line.rstrip('\n') for line in file]
    for line in lines:
        w = line.split()
        x = [w[0], w[1] + " " + w[2] + " " + w[3] + " " + w[4], w[5] + " " + w[6] + " " + w[7] + " " + w[8],
             w[9] + " " + w[10] + " " + w[11] + " " + w[12]]
        splitArray.append(x)
    print(splitArray)
    return splitArray


def main():
    splitArray = checkSetUp()
    mainLoop(splitArray)


if __name__ == "__main__":
    main()
