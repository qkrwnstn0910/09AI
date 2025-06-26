from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout, Flatten, Dense
import numpy as np , os
from PIL import Image

categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

image_w = 64
image_h = 64

data = np.load("./saveFiles/caltech_5object.npz")
x_train = data["x_train"]
x_test = data["x_test"]
y_train = data["y_train"]
y_test = data["y_test"]

x_train = x_train.astype("float") / 256
x_test = x_test.astype("float") / 256

print('X_train shape', x_train.shape)

model = Sequential()

model.add(Conv2D(32, (3,3), padding ='same', input_shape =x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

hdf5_file = "./saveFiles/caltech_5object.model.hdf5"
if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
else:
    model.fit(x_train, y_train, batch_size=32, epochs=50)
    model.save_weights(hdf5_file)

pre = model.predict(x_test)
for i , v in enumerate(pre):
    pre_ans = v.argmax()
    ans = y_test[i].argmax()
    dat = x_test[i]
    if ans == pre_ans: continue

    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)
    fname = "./predict_error/" + str(i) + "-" + categories[pre_ans] +\
        "-ne-" + categories[ans] + ".png"

    dat *=256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)

score = model.evaluate(x_test, y_test)
print('loss', score[0])
print('accuracy', score[1])
