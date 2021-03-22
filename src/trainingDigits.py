#!/usr/bin/env python3

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

imageDimensions = (32, 32, 3)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR TO GRAY IMAGE
    img = cv2.equalizeHist(img)  # FIXING THE LIGHT
    img = img/255
    return img


def setUpGlobals():
    path = 'myData'
    images = []  # LIST CONTAINING ALL THE IMAGES
    classNo = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
    myList = os.listdir(path)
    noOfClasses = len(myList)
    print("Total Classes Detected:", noOfClasses)
    print("Importing Classes .......")
    for x in range(0, noOfClasses):
        myPicList = os.listdir(path + "/" + str(x))
        for y in myPicList:
            curImg = cv2.imread(path + "/" + str(x) + "/" + y)
            curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))  # RESIZING IMAGE TO MAKE QUICKER THE LEARNING
            images.append(curImg)
            classNo.append(x)
        print(x, end=" ")
    print(" ")
    print("Total Images in Images List = ", len(images))
    print("Total IDS in classNo List= ", len(classNo))
    print("")
    return np.array(images), np.array(classNo), noOfClasses


def splitData(images, classNo, noOfClasses, debug):
    testRatio = 0.2
    valRatio = 0.2
    X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
    print("80% for train : ", X_train.shape)
    print("20% for test : ", X_test.shape)
    print("20% of 80% in train for validation : ", X_validation.shape)
    print("")
    print("")

    numOfImagesForId = []
    for x in range(0, noOfClasses):
        print("There are ", len(np.where(y_train == x)[0]), "images for number", x, "in training (80%)")
        numOfImagesForId.append(len(np.where(y_train == x)[0]))
    print("In array :", numOfImagesForId)

    if debug:
        plt.figure(figsize=(10, 5))
        plt.bar(range(0, noOfClasses), numOfImagesForId)
        plt.title("No of Images for each Class")
        plt.xlabel("Class ID")
        plt.ylabel("Number of Images")
        plt.show()

    if debug:
        img = X_train[30]
        img = cv2.resize(img, (300, 300))
        cv2.imshow("PreProcesssed", img)
        cv2.waitKey(0)

        img2 = preProcessing(X_train[30])
        img2 = cv2.resize(img2, (300, 300))
        cv2.imshow("PreProcesssed", img2)
        cv2.waitKey(0)

    return X_train, X_test, X_validation, y_train, y_test, y_validation


def processAllImages(X_train, X_test, X_validation):
    X_train = np.array(list(map(preProcessing, X_train)))
    X_test = np.array(list(map(preProcessing, X_test)))
    X_validation = np.array(list(map(preProcessing, X_validation)))

    return X_train, X_test, X_validation


def addDepth(X_train, X_test, X_validation):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # RESHAPING THE ARRAY ADDING A DEPTH FOR AFTER
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

    return X_train, X_test, X_validation


def generateImageModification(X_train):
    #### IMAGE MODIFICATION, rotation, zoom, cut, shift ... to train with more and more images
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)
    dataGen.fit(X_train)

    return dataGen


def oneHotEncodeMatrice(y_train, y_test, y_validation, noOfClasses):
    #### ONE HOT ENCODING OF MATRICES
    # A binary matrix representation of the input. The classes axis is placed last.
    #
    # Exemple :
    #
    # a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    # a = tf.constant(a, shape=[4, 4])
    # print(a)
    #
    # Printed :
    #
    # tf.Tensor(
    # [[1. 0. 0. 0.]
    #  [0. 1. 0. 0.]
    #  [0. 0. 1. 0.]
    #  [0. 0. 0. 1.]], shape=(4,4), dtype=float32)

    y_train = to_categorical(y_train, noOfClasses)
    y_test = to_categorical(y_test, noOfClasses)
    y_validation = to_categorical(y_validation, noOfClasses)

    return y_train, y_test, y_validation


def myModel(noOfClasses):

    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def startTraining(model, dataGen, X_train, X_validation, y_train, y_validation):
    #### STARTING THE TRAINING PROCESS

    batchSizeVal = 50
    epochsVal = 10
    stepsPerEpochVal = 2000

    history = model.fit_generator(dataGen.flow(X_train, y_train,
                                               batch_size=batchSizeVal),
                                  steps_per_epoch=stepsPerEpochVal,
                                  epochs=epochsVal,
                                  validation_data=(X_validation, y_validation),
                                  shuffle=1)

    #### PLOT THE RESULTS
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()


def evaluate(model, X_test, y_test):
    #### EVALUATE USING TEST IMAGES

    score = model.evaluate(X_test, y_test, verbose=0)
    print("")
    print('Test Score = ', score[0])
    print('Test Accuracy =', score[1])


def saveModel(model):
    #### SAVE THE TRAINED MODEL

    pickle_out = open("model_trained.p", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def main():
    debug = False
    if len(sys.argv) >= 2 and sys.argv[1] == "-d":
        debug = True

    images, classNo, noOfClasses = setUpGlobals()
    print("Shape images : ", images.shape)
    print("Shape classNo : ", classNo.shape)
    X_train, X_test, X_validation, y_train, y_test, y_validation = splitData(images, classNo, noOfClasses, debug)
    X_train, X_test, X_validation = processAllImages(X_train, X_test, X_validation)
    X_train, X_test, X_validation = addDepth(X_train, X_test, X_validation)
    dataGen = generateImageModification(X_train)
    y_train, y_test, y_validation = oneHotEncodeMatrice(y_train, y_test, y_validation, noOfClasses)
    model = myModel(noOfClasses)
    startTraining(model, dataGen, X_train, X_validation, y_train, y_validation)
    evaluate(model, X_test, y_test)
    saveModel(model)


if __name__ == "__main__":
    # execute only if run as a script
    main()
