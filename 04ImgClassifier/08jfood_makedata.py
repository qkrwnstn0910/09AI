from sklearn.model_selection import train_test_split
from PIL import Image
import glob as gl
import numpy as np

root_dir = "./download"
categories = ["Gyudon","Ramen", "Sushi", "Okonomiyaki", "Karaage"]

nb_classes = len(categories)

image_size = 50

X = []
Y = []

for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = gl.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)

        X.append(data)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y)

np.savez("./saveFiles/japanese_food.npz", x_train=x_train, x_test=x_test
         ,y_train=y_train, y_test=y_test)
print("Task Finished..!!", len(Y))

data = np.load("./saveFiles/japanese_food.npz")
print("x_train shape:", data["x_train"].shape)
print("y_train shape:", data["y_train"].shape)
print("x_test shape:", data["x_test"].shape)
print("y_test shape:", data["y_test"].shape)
