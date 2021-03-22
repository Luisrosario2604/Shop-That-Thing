#!/usr/bin/env python3.7

import numpy as np
import cv2 as cv
import pickle
import sys
from gtts import gTTS
import os


capture = None
detectPercentage = 65


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def speek(objectNumber, splitArray, codeWhatAppend):
    square = splitArray[objectNumber]
    if codeWhatAppend == 1:
        sentence = "Vous avez pris " + square[0]
    elif codeWhatAppend == 2:
        sentence = "Vous avez repose " + square[0]
    print(sentence)
    #audio = gTTS(text=sentence, lang='fr', slow=False)
    #audio.save("sentence.mp3")
    #os.system("mpg321 sentence.mp3")


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
    mask = cv.inRange(hsv, (85, 30, 20), (130, 255, 255))
    cv.imshow(name + " colored", mask)
    percentageDetected = (mask > 0).mean()
    #print("Detected =>" + to_str(percentageDetected))

    return percentageDetected * 100


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


def mainLoop(splitArray, totalObjects):
    checkArgumentsForCamera()        

    pickle_in = open("model_trained_10.p", "rb")
    model = pickle.load(pickle_in)
    objectTaken = ["0"] * totalObjects

    while True:
        _, imgOriginal = capture.read()
        cv.imshow("Original Image", imgOriginal)

        #nm, proba = numberFromImage(imgOriginal, model, 0, splitArray)
        #if proba > 0.90:
        #    print(nm, proba)

        for objectNumber in range(0, totalObjects):
            percentage = squareFromImage(imgOriginal, objectNumber, splitArray)
            if float(objectTaken[objectNumber]) < detectPercentage <= percentage:
                speek(objectNumber, splitArray, 1)
            if float(objectTaken[objectNumber]) > detectPercentage >= percentage:
                speek(objectNumber, splitArray, 2)
            objectTaken[objectNumber] = to_str(np.around(percentage, 2))

        #print("Object Taken => " + ', '.join(objectTaken))

        #seeCropped(imgOriginal, 0, splitArray)
        if cv.waitKey(33) and 0xFF == ord('q'):
            break


def checkSetUp():
    splitArray = []
    file = open('dataSquares.txt', 'r')
    lines = [line.rstrip('\n') for line in file]
    totalObjects = 0
    for line in lines:
        w = line.split()
        x = [w[0], w[1] + " " + w[2] + " " + w[3] + " " + w[4], w[5] + " " + w[6] + " " + w[7] + " " + w[8],
             w[9] + " " + w[10] + " " + w[11] + " " + w[12]]
        splitArray.append(x)
        totalObjects += 1
    print(splitArray)
    return splitArray, totalObjects


def main():
    splitArray, totalObjects = checkSetUp()
    mainLoop(splitArray, totalObjects)


if __name__ == "__main__":
    main()
