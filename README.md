# ShopThatThing (Machine Learning + Color and Shape detection)

The goal of the project is to create a prototype of an automated store.

## Goals

- Detection of objects taken
- The camera must be managed by the webcam or an android phone
- Price detection by machine learning (digits machine learning)
- Color and shape detection

## Requirements

* Python 3.7+

* gTTS >= 2.0.3
* opencv_python >= 4.2.0.34
* Keras >= 2.3.1
* scikit_learn >= 0.24.1
* numpy >= 1.18.1
* sklearn >= 0.0
* matplotlib >= 3.3.3
* tensorflow >= 2.0.0

How to install all the requirements :
```bash
sudo pip3 install -r requirements.txt
```

## Usage

### Digit recognition by machine learning

If you want to try the trained model for digit recognition :
This will help to detect the price on top of each item.

If you want to test it with you webcam :

```bash
cd src/
./checkModel.py
```

If you want to test it with your android phone (download "IP webcam" app) :

```bash
cd src/
./checkModel.py [IP of the camera + /video]

Example :
cd src/
./checkModel.py https://192.168.43.1:8080/video
```

![digit](./gif/digit.gif)


### Shop and item detection

For this part you will need to "build" a little shop. You will need to put a blue square every single item that you want to sell. This will help for the item detection.

#### Setup

You will also need to setup your shop :
Here you will have to setup all the stands of the shop, showing where the object is, its name and where its price is.

```bash
cd src/
./setUp.py

or

cd src/
./setUp.py +[phone wifi ip]/video
```
###### This will generate the dataSquares.txt file saving all the stands positions
###### Press "n" key to create a new position and follow cmd instructions

#### Shop by color detection

This will detect if an item as been taken or not. It works with the blue squares on the back of each item (detecting if the blue color is there at 90%).

```bash
cd src/
./checkMultiModel.py

or

cd src/
./checkMultiModel.py +[phone wifi ip]/video
```
![color](./gif/color.gif)

#### Shop by shape detection

This will detect if an item as been taken or not. It works with the blue squares on the back of each item (detecting if the square is there).

```bash
cd src/
./checkMultiSquare.py

or

cd src/
./checkMultiSquare.py +[phone wifi ip]/video
```
![square](./gif/square.gif)
## Authors

* **Luis Rosario** - *Initial work* - [Luisrosario](https://github.com/Luisrosario2604)