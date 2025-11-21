import cv2
import numpy as np

img_path = "../imgs/ocr_test6.jpg"

# 1. Morphology 기반 그림자 제거
#      각 채널(R, G, B)에 대해
#      큰 커널로 배경(조명) 추정 후
#      원본에서 빼서 그림자를 줄이는 방식
def Morphology():
    img = cv2.imread(img_path)

    # 채널 분리 (B, G, R 순서)
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        # 팽창 연산으로 배경(조명) 부분을 크게 확장해서 추정
        # kernel: 구조 요소(커널) 크기와 모양
        dilated = cv2.dilate(
            plane,
            np.ones((16, 16), np.uint8)  # 20x20 커널, 타입은 uint8
        )

        # 배경 블러 처리로 부드럽게 만들기
        # ksize: 커널 크기(홀수), 값이 클수록 더 많이 블러됨
        bg = cv2.medianBlur(dilated, 37)

        # 그림자 제거
        # 두 이미지의 절대 차이를 계산(255 - absdiff)해서 밝기를 반전하는 효과
        diff = 255 - cv2.absdiff(plane, bg)

        # 채널별 결과 저장
        result_planes.append(diff)

    # 채널 합치기
    result = cv2.merge(result_planes)
    return result

# Morphology 결과 저장
out = Morphology()
cv2.imwrite("../imgs/Morphology.png", out)

# 2. Background Subtraction
#      그레이스케일로 만든 뒤 블러로 배경(조명) 추정 후 나누기 연산으로 조명을 평탄화
#      이후 CLAHE로 대비 강화
def BackgroundSubtraction():
    img = cv2.imread(img_path)

    # BGR 에서 GRAY 로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 배경 추정을 위한 미디언 블러
    # 커널 크기가 클수록 전체적인 조명만 남고 디테일이 사라짐
    bg = cv2.medianBlur(gray, 61)

    # 조명 보정: gray / bg
    # scale: 결과에 곱해줄 스케일
    norm = cv2.divide(gray, bg, scale=255)

    # 대비(콘트라스트) 강화
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit: 대비 제한 정도(클수록 대비 더 세게)
    # tileGridSize: 영역을 얼마나 쪼개서 적용할지 (가로, 세로 분할 수)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    # clahe 적용(norm은 그레이스케일)
    enhanced = clahe.apply(norm)
    return enhanced

out = BackgroundSubtraction()
cv2.imwrite("../imgs/BackgroundSubtraction.png", out)

# 3. Adaptive Threshold
#      그림자나 조명 차이를 무시하고
#      국소 영역 기준으로 이진화

# 이미지 읽기 (그레이스케일)
img = cv2.imread(img_path, 0)

# 적응형 이진화
# maxValue: 임계값을 넘을 때 줄 값 (보통 255)
# adaptiveMethod:
#   cv2.ADAPTIVE_THRESH_MEAN_C    주변 평균 사용
#   cv2.ADAPTIVE_THRESH_GAUSSIAN_C 주변 가중 평균(가우시안) 사용
# thresholdType:
#   cv2.THRESH_BINARY             (픽셀값 > 임계값 이면 maxValue, 아니면 0)
# blockSize: 임계값 계산할 지역 크기(홀수)
# C: 계산된 평균에서 얼마나 빼줄지 (값이 클수록 더 어두운 픽셀만 흰색으로)
out = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    16
)

cv2.imwrite("../imgs/AdaptiveThreshold.png", out)

# 4. Gray World 스타일 조명 보정
#      가우시안 블러로 조명 성분 추정 후
#      원본 밝기를 조명으로 나눠서 평탄화

img = cv2.imread(img_path)

# BGR 에서 GRAY 로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 가우시안 블러로 조명(illumination) 추정
# ksize: 커널 크기 (가로, 세로, 둘 다 홀수)
# sigmaX: 가우시안 커널의 표준편차 (0이면 자동 계산)
# sigmaY: 0이면 sigmaX 와 동일하게 사용
ksize = 151
illum = cv2.GaussianBlur(gray, (ksize, ksize), 0)

# divide 연산으로 어두운 부분(그림자) 비율을 줄이고 밝기 균일화
norm = cv2.divide(gray, illum, scale=255)

cv2.imwrite("../imgs/GrayWorld.png", norm)
