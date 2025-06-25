from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

csv = pd.read_csv("./resData/bmi.csv")

# 몸무게와 키 데이터를 0~1사이의 값을 정규화한다.
csv["height"] /= 200
csv[" weight"] /= 100
# 입력데이터로 사용 정규화된 몸무게와 키 컬럼을추출함
X=csv[[" weight", "height"]]

# 레이블 변환 원-핫 인코딩 형식의 딕셔너리 생성
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat":[0,0,1]}
# 2만개의 레이블을 bclass 형식으로 변환
y=np.empty((20000,3))
for i, v in enumerate(csv["label"]):
    y[i] = bclass[v]

X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test = X[15001:20001], y[15001:20001]

# 배열 형식으로 레이어 정의 후 모델 구조 정의
# 각 층을 리스트에 추가하는 형식으로 정의(add 할수 사용과 동일)
layers = [
    Dense(512, input_shape=(2,)),
    Activation('relu'),
    Dropout(0.1),
    Dense(512),
    Activation('relu'),
    Dropout(0.1),
    Dense(3),
    Activation('softmax')
]
model = Sequential(layers)

# 모델 구축
model.compile(
    loss = 'categorical_crossentropy',
    optimizer="rmsprop",
    metrics =["accuracy"]

)
hist = model.fit(
    X_train, y_train,
    batch_size=100, #100개의 샘플씩 묶어 한번에 학습
    epochs=20, #최대 2개번의 반복 학습
    validation_split=0.1, #훈련 데이터 중 10%를 검증 데이터로 사용
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1)
'''
moniter='var_loss' : 검증 손실값(val_loss)을 모니터링 함
patience = 2 : 검증 손실값이 개선되지 않으면 학습을 2번 더 진행한 후 조기종료'''

score = model.evaluate(X_test, y_test)
print('loss=',score[0])
print('accuracy=', score[1])