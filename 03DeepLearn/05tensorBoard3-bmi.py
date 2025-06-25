import tensorflow as tf
import pandas as pd
import numpy as np

#데이터 로드 후 데이터 프레임으로 변환
csv = pd.read_csv("./resData/bmi.csv")
print(csv.columns)
# 데이터 정규화 후 각 컬럼에 즉시 적용
csv["height"] = csv["height"] / 200
csv[" weight"] = csv[" weight"] / 100

bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x]))

test_csv = csv[15000:20000]
test_pat = np.array(test_csv[[" weight", "height"]])
test_ans = np.array(list(test_csv["label_pat"]))

'''입력레이어 : 키와 몸무개가 2개의 입력값을 사용
   출력레이어 : 3개의 클래스(thin, normal, fat)사용 활성화 함수는 softmax 사용'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_pat = np.array(csv[[" weight","height"]])
train_ans = np.array(list(csv["label_pat"]))

import datetime as da
log_dir = "log_dir/"+ da.datetime.now().strftime("%Y%m%d-%H$M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
'''
epochs : 데이터셋 전체를 35번 반복 학습
batch_size : 학습 중 한번에 사용할 데이터의 크기 지정
validation_data : 테스트 데이터를 사용해 매 epochs마다 모델의 성능을 검증
verbose : 훈련진행 상황을 시각적으로 표시'''
history = model.fit(
    train_pat, train_ans,
    epochs=35,
    batch_size=100,
    validation_data=(test_pat, test_ans),
    verbose=1,
    callbacks=[tensorboard_callback]
)
test_loss, test_acc= model.evaluate(test_pat, test_ans)
print("정답률", test_acc)

with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.scalar("Test Accuracy", test_acc, step=0)
    tf.summary.scalar("Test Loss", test_loss, step=0)

print(f"TensorBoard write ok : {log_dir}")