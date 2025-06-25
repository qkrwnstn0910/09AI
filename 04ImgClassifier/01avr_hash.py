from PIL import Image
import numpy as np

# 이미지 데이터를 Average hash로 전환
def average_hash(fname, size = 16):
    # 이미지 오픈
    img = Image.open(fname)
    # 그레이 스케일로 변환(흑백조)
    img = img.convert('L')
    # 리사이즈 16*16 크기 Lanczos 리샘플링 알고리즘 적용
    img = img.resize((size,size), Image.Resampling.LANCZOS)
    # 이미지 픽셀 데이터를 1차원 리스트로 가져옴
    pixel_data = img.getdata()
    # 픽셀 데이터를 numpy 배열로 변환
    pixels = np.array(pixel_data)
    # 1차원을 2차원으로 변환
    pixels = pixels.reshape((size, size))
    # 전체 픽셀의 평균값 계산
    avg = pixels.mean()
    # 평균보다 큰 픽셀은 1 작거나 같으면 0으로 변환
    diff = 1 * (pixels > avg)

    return diff
# 이진 배열을 16진수 문자열 해시로 변환
def bp2hash(ahash):
    bhash = []
    # 2차원 numpy배열을 리스트로 변환 후 반복
    for n1 in ahash.tolist():
        # 각 행의 원소를 문자열로 변환
        s1 = [str(i) for i in n1]
        # 문자열을 하나로 이어붙힘
        s2 = "".join(s1)
        # 2진 문자열을 10진 정수로 변환
        i = int(s2, 2)
        # 리스트에 16진수로 변환 후 추가한다.
        bhash.append("%04x" % i)
    return "".join(bhash)

ahash = average_hash('./resData/tower.jpg')
print("출력1\n", ahash)
print("출력2\n", bp2hash(ahash))
