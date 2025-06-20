import numpy as np

'''
ndarray에 저장하는 데이터는 숫자 문자열 불리안 등 모두 가능하다.
단 연산의 특성상 같은 자료형이어야 한다.'''

# 정수만으로 리스트 생성
list1 = [1,2,3]
# 리스트를 ndarray로 변환
array1 = np.array(list1)
print('array1 출력:', array1)
#배열의 타입 확인 : ndarray
print('array1 타입:', type(array1))
#배열에 저장된 데이터 타입 확인 : int32
print('array1 dtype', array1.dtype)

# 실수가 포함된 리스트 생성
list2 = [1,2,3.0]
# 배열로 변환시 모든 데이터가 실수로 변환된다.
array2 = np.array(list2)

print('array2 출력:', array2)
print('array2 타입:', type(array2))
# 실수가 포함되었으므로 float64로 출력된다.
print('array2 dtype', array2.dtype)


list3 = [1,2,'test']
array3 = np.array(list3)

print('array3 출력:', array3)
print('array3 타입:', type(array3))
print('array3 dtype', array3.dtype)

array_int1 = np.array([1,2,3])
array_float1 = array_int1.astype('float64')
print(array_float1, array_float1.dtype)

array_float2 = np.array([1.1,2.1,3.1])
array_int2 = array_float2.astype('int32')
print(array_int2,array_int2.dtype)
