from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D as Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense
from tensorflow.keras import utils
import numpy as np

# root_dir = "./download"
categories = ["Gyudon","Ramen", "Sushi", "Okonomiyaki", "Karaage"]
nb_classes = len(categories)
image_size = 100

def main():
    data = np.load("./saveFiles/japanese_food_aug.npz")
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256

    y_train = utils.to_categorical(y_train, nb_classes)
    y_test = utils.to_categorical(y_test, nb_classes)

    model = model_train(x_train, y_train)
    model_eval(model, x_test, y_test)

def build_model(in_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def model_train(X,Y):
    model = build_model(X.shape[1:])
    model.fit(X,Y,batch_size=32, epochs=30)
    hdf5_file = "./saveFiles/japanese_food_aug_model.hdf5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model,X,Y):
    score = model.evaluate(X,Y)
    print('loss', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()