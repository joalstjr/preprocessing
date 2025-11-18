import cv2
import numpy as np


##### Morphology #####
def Morphology(img_path):
    img = cv2.imread(img_path)
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        # 큰 커널로 배경(조명) 추정
        dilated = cv2.dilate(plane, np.ones((20, 20), np.uint8))
        bg = cv2.medianBlur(dilated, 15)

        # 그림자 제거: 원본 - 배경
        diff = 255 - cv2.absdiff(plane, bg)
        result_planes.append(diff)

    result = cv2.merge(result_planes)
    return result
out = Morphology("shadow_test1.png")
cv2.imwrite("Morphology.png", out)


##### Background Subtraction #####
def BackgroundSubtraction(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##### Background Normalization #####
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)

    # 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(norm)
    return enhanced
out = BackgroundSubtraction("shadow_test1.png")
cv2.imwrite("BackgroundSubtraction.png", out)


##### Adaptive Threshold로 그림자 무시 후 이진화 #####
img = cv2.imread("shadow_test1.png", 0)
out = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51, 15
)
cv2.imwrite("AdaptiveThreshold.png", out)


##### Gray World(조명보정 + 가우시안 블러) #####
img = cv2.imread("shadow_test1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 조명 보정용 블러
illum = cv2.GaussianBlur(gray, (55,55), 0)
norm = cv2.divide(gray, illum, scale=255)

cv2.imwrite("GrayWorld.png", norm)