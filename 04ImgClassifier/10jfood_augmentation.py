from PIL import Image, ImageEnhance
import glob
import numpy as np

root_dir = "./download"
categories = ["Gyudon","Ramen", "Sushi", "Okonomiyaki", "Karaage"]
nb_classes = len(categories)
image_size = 100

X = []
Y = []

def add_sample(cat, fname, is_train):

    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size,image_size))
    data = np.asarray(img)

    X.append(data)
    Y.append(cat)

    if not is_train:
        return

    for ang in range(-30, 30, 10):
        img2 = img.rotate(ang)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)

def make_sample(files, is_train):
    global X, Y
    X = []; Y=[]
    for cat, fname in files:
        add_sample(cat,fname,is_train)
    return np.array(X), np.array(Y)

allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    print("---", cat,"처리중")
    for f in files:
        print(f)
        allfiles.append((idx, f))

from sklearn.model_selection import train_test_split
train, test = train_test_split(allfiles, test_size=0.3,
                                          stratify=[x[0]for x in allfiles],
                                          random_state=42)
x_train,y_train = make_sample(train,True
)
x_test, y_test = make_sample(test,False)

np.savez("./saveFiles/japanese_food_aug.npz", x_train=x_train,x_test=x_test,
         y_train=y_train, y_test=y_test)
print("Task Finished",len(y_train))