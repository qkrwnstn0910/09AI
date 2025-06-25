from pandas.core.computation.expressions import evaluate
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#e데이터를 float형태로 바꾸고 정규화한다.
# 28*28 크기의 2차원 이미지를 784차원의 1차원 배열로 변환하고 훈련용 6만개
# 테스트용 1만개로 나는다
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train /= 255
X_test /= 255

# 정수형 레이블 데이터를 0~9까지의 카테고리를 나타내는 배열로 변환
# 즉 원 핫 인코딩 형태로 변환한다.
y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test, 10)

#Sequential() : 모델과 각 레이어를 순차적으로 추가하는 방식의 신경망 구성
model = Sequential()
#dense : 완전연결층으로 n개의 뉴런을 가진다 입력데이터는 784차원으로 설정
model.add(Dense(512, input_shape=(784,)))
# 활성화함수는 relu로 설정
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
# 여기까지 은닉층 다음부터는 출력층으로 구성됨 즉 2개의 은닉층과 1개의 출력층
# 10개의 뉴런 및 각 클래스에 속할 확률을 계산한
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
# 모델구축
'''
손실함수 : 크로스 엔트로피
옵티마이저 : 아담
학습중 모델의 정확도를 측정하도록 설정'''
model.compile(
    loss = 'categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
hist = model.fit(X_train, y_train)

score = model.evaluate(X_test, y_test, verbose=1)
print('loss=', score[0])
print('accuracy=', score[1])