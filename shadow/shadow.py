import cv2
import numpy as np


# ==============================
# 1. Morphology 기반 그림자 제거
#    - 각 채널(R, G, B)에 대해
#      큰 커널로 배경(조명) 추정 후
#      원본에서 빼서 그림자를 줄이는 방식
# ==============================
def Morphology(img_path):
    # 이미지 읽기
    # cv2.imread(filename, flags)
    # filename: 이미지 경로
    # flags: 이미지 읽기 방식 (기본값은 컬러 BGR)
    img = cv2.imread(img_path)

    # 채널 분리 (B, G, R 순서)
    # cv2.split(m)
    # m: 다채널 이미지
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        # 팽창 연산으로 배경(조명) 부분을 크게 확장해서 추정
        # cv2.dilate(src, kernel, iterations)
        # src: 입력 이미지(단일 채널)
        # kernel: 구조 요소(커널) 크기와 모양
        #         여기서는 20x20 크기의 사각형 커널 사용
        # iterations: 반복 횟수 (기본 1, 생략)
        dilated = cv2.dilate(
            plane,
            np.ones((20, 20), np.uint8)  # 20x20 커널, 타입은 uint8
        )

        # 배경 블러 처리로 부드럽게 만들기
        # cv2.medianBlur(src, ksize)
        # src: 입력 이미지
        # ksize: 커널 크기(홀수), 값이 클수록 더 많이 블러됨
        bg = cv2.medianBlur(dilated, 15)

        # 그림자 제거
        # cv2.absdiff(src1, src2)
        # 두 이미지의 절대 차이 계산
        # 255 - absdiff 를 해서 밝기를 반전하는 효과
        diff = 255 - cv2.absdiff(plane, bg)

        result_planes.append(diff)

    # 다시 채널 합치기
    # cv2.merge(mv)
    # mv: 합칠 채널 리스트
    result = cv2.merge(result_planes)
    return result


# Morphology 결과 저장
out = Morphology("shadow_test1.png")
cv2.imwrite("Morphology.png", out)



# ==============================
# 2. Background Subtraction
#    - 그레이스케일로 만든 뒤
#      블러로 배경(조명) 추정 후 나누기 연산으로
#      조명을 평탄화
#    - 이후 CLAHE로 대비 강화
# ==============================
def BackgroundSubtraction(img_path):
    # 이미지 읽기 (컬러)
    img = cv2.imread(img_path)

    # BGR 에서 GRAY 로 변환
    # cv2.cvtColor(src, code)
    # src: 입력 이미지
    # code: 색 공간 변환 코드
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 배경 추정을 위한 미디언 블러
    # 커널 크기가 클수록 전체적인 조명만 남고 디테일이 사라짐
    bg = cv2.medianBlur(gray, 31)

    # 조명 보정: gray / bg
    # cv2.divide(src1, src2, dst, scale, dtype)
    # src1: 분자 이미지
    # src2: 분모 이미지
    # scale: 결과에 곱해줄 스케일(여기서는 255)
    norm = cv2.divide(gray, bg, scale=255)

    # 대비(콘트라스트) 강화
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # cv2.createCLAHE(clipLimit, tileGridSize)
    # clipLimit: 대비 제한 정도(클수록 대비 더 세게)
    # tileGridSize: 영역을 얼마나 쪼개서 적용할지 (가로, 세로 분할 수)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # clahe.apply(src)
    # src: 그레이스케일 이미지
    enhanced = clahe.apply(norm)
    return enhanced


# Background Subtraction 결과 저장
out = BackgroundSubtraction("shadow_test1.png")
cv2.imwrite("BackgroundSubtraction.png", out)



# ==============================
# 3. Adaptive Threshold
#    - 그림자나 조명 차이를 무시하고
#      국소 영역 기준으로 이진화
# ==============================
# 이미지 읽기 (그레이스케일)
# cv2.imread(filename, flags)
# flags = 0 이면 그레이스케일로 읽기
img = cv2.imread("shadow_test1.png", 0)

# 적응형 이진화
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod,
#                       thresholdType, blockSize, C)
# src: 입력 그레이스케일 이미지
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
    15
)

cv2.imwrite("AdaptiveThreshold.png", out)



# ==============================
# 4. Gray World 스타일 조명 보정
#    - 가우시안 블러로 조명 성분 추정 후
#      원본 밝기를 조명으로 나눠서 평탄화
# ==============================
# 이미지 읽기 (컬러)
img = cv2.imread("shadow_test1.png")

# BGR 에서 GRAY 로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 가우시안 블러로 조명(illumination) 추정
# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
# src: 입력 이미지
# ksize: 커널 크기 (가로, 세로, 둘 다 홀수)
# sigmaX: 가우시안 커널의 표준편차 (0이면 자동 계산)
# sigmaY: 0이면 sigmaX 와 동일하게 사용
illum = cv2.GaussianBlur(gray, (55, 55), 0)

# 조명 보정: gray / illum
# divide 연산으로 어두운 부분(그림자) 비율을 줄이고 밝기 균일화
norm = cv2.divide(gray, illum, scale=255)

cv2.imwrite("GrayWorld.png", norm)
