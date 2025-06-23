import urllib.request as req

local = "./reData/mushroom.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
req.urltrieve(url, local)
print("ok")