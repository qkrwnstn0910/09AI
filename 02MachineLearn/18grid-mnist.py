import pandas as pd
from sklearn import svm,metrics
from sklearn.model_selection import GridSearchCV

train_csv = pd.read_csv("resMnist/train.csv", header=None)
test_csv = pd.read_csv("resMnist/t10k.csv", headet=None)
print(train_csv)

# 필요한 열 추출
'''첫번째 열은 라벨, 두번째부터는 특성 데이터이므로 추출하여 저장한다.'''
train_label = train_csv.iloc[:,0]
train_data = train_csv.iloc[:,1:577]
test_label = test_csv.iloc[:,0]
test_data = test_csv.iloc[:,1:577]
print("학습 데이터의 수", len(train_label))

# 그리드 서치 매개변수 설정
'''
svm의 다양한 매개변수 조합을 설정할 수 있다.
C : 정규화강도
    설정한 강도에 따라서 얼만큼 완벽한 분리를 할 수 있을지를 결정할 수 있다.
kernel : 커널함수의 종류
    데이터를 어떻게 분류할 것인가를 결정한다.(linear, rbf, poly등)
gamma : RBF 커널의 감마 값
    데이터의 패턴을분석 할 수 있다.
'''
params = [
    {"C": [1,10,100,1000], "kernel":["linear"]},
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0,0.01,0.001]}
]

# 그리드 서치 수행 : 설정된 매개변수 조합 중 최적의 값을 찾음
clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)
clf.fit(train_data,train_label)
# 학습 데이터로 최적의 모델을 학습한다
print("학습기 =",clf.best_estimator_)
print("-----------------")
print(train_data.columns)
print(test_data.columns)
print('------------------')

pre = clf.predict(test.data)
ac_score = metrics.accuracy_score(pre, test_label)
print("정답률",ac_score)