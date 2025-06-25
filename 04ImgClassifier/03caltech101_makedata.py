from PIL import  Image
import os, glob
import  numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = "./caltech101/101_ObjectCategories"
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

image_w = 64
image_h = 64

pixels = image_w * image_h *3

x= []
y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + "/"+cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w,image_h))

        data = np.asarray(img)

        x.append(data)
        y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

np.savez("./saveFiles/caltech_5object.npz", x_train=x_train, x_test=x_test,
         y_train=y_train, y_test=y_test)
print("Task Finished", len(y))