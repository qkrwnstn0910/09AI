import numpy as np

A = np.array([1,2,3])
B = 5
result = A+B
print('결과1',result)
# 2행 3열과 1행 3열의 배열 생성
A=np.array([[1,2,3],[4,5,6]])
B=np.array([1,2,3])
# 2행 3열로 확장된 후 계산 수행
result = A+B
print('결과2\n', result)

# 2행3열과 1행 2열의 배열 생성
A=np.array([[1,2,3],[4,5,6]])
B=np.array([1,2])
'''오류발생
두 배열의 마지막 차원의 크기가3과 2로 다르고 둘중 하나만 1이 아니므로 
브로드캐스팅이 되지 않는다.
'''
# result = A+B
# 3행 3열인 2차원 배열 생성
C = np.arange(9.).reshape(3,3)
# 1차원 배열 생성
x = np.array([1,0,0]) #전체가 실수로 변환됨
# 배열 x를 1행 3열 3행 1열인 2차원 배열로 변환
y= x.reshape(1,3)
z= x.reshape(3,1)
# 2차원배열 + 스칼라 : 스칼라가 배열로 확장되어 연산
result = C+10
print('결과3\n',result)

# 2차원배열 +1차원배열 : 1치원배열이 브로드캐스팅되어 연산
result = C+x
print('결과4\n',result)

result = C+y
print('결과5\n',result)

result = C+z
print('결과6\n',result)

result = y+z
print('7\n',result)