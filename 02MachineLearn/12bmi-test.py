from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import pandas as pd

tbl = pd.read_csv("resData/bmi.csv")
label = tbl["label"]
w = tbl["weight"]
h = tbl["height"]

wh = pd.concat([w,h], axis=1)

data_train, data_test, label_train, label_test = train_test_split(wh.label)

clf = svm.SVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률=", ac_score)
print("리포트")
print(cl_report)