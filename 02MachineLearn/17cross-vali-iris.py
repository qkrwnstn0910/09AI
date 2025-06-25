import pandas as pd
from sklearn import svm,model_selection

csv = pd.read_csv('resData/iris.csv')

data = csv[["sepalLength","sepalWidth", "petalLength", "petalWidth"]]
label = csv["name"]
'''
cross_val_score()함수는 교차 검증을 수행한다.
cv옵션에 지정한 수 만큼 데이터를 분할하여 학습과 테스트를 반복한다. 즉 여기서는 
5등분(folds)해서 5번 학습과 테스트를 진행하겠다는 의미'''
clf = svm.SVC()
scores = model_selection.cross_val_score(clf,data,label, cv=5)

print("각각의 정답률=",scores)
print("평균 정답률",scores.mean())