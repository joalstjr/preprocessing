import cv2
import numpy as np


def order_points(pts):
    """
    사각형 네 점을
    [좌상, 우상, 우하, 좌하] 순서로 정렬
    pts: (4, 2) float32 배열
    """
    rect = np.zeros((4, 2), dtype="float32")

    # x + y 가 가장 작은 점이 좌상, 가장 큰 점이 우하
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # x - y 가 가장 작은 점이 우상, 가장 큰 점이 좌하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    네 꼭짓점 좌표로 퍼스펙티브 변환 수행
    단일 문서 한 장을 평평하게 펴는 용도
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 새 이미지의 폭 계산
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # 새 이미지의 높이 계산
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # 출력 이미지에서의 네 점 위치
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 변환 행렬 계산 후 워핑
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_document_and_warp(img_path, debug_prefix=None):
    """
    단일 문서 사진에서
    1. 문서 전체 윤곽(사각형) 검출
    2. 퍼스펙티브 보정으로 문서를 평평하게 펴기

    Canny 대신 Otsu 이진화 기반으로 문서 전체를 검출
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 못 찾음: {img_path}")

    orig = image.copy()

    # 너무 크면 속도 느리니까 높이를 700 정도로 줄여서 처리
    target_height = 700
    ratio = image.shape[0] / float(target_height)
    new_height = target_height
    new_width = int(image.shape[1] / ratio)
    image = cv2.resize(image, (new_width, new_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 이진화로 문서 영역을 덩어리로 만들기
    # 배경(책상, 담요 등)보다 종이가 더 밝다고 가정
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 약간 팽창 + 닫기 연산으로 구멍 메우기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_gray.png", gray)
        cv2.imwrite(f"{debug_prefix}_thresh.png", thresh)
        cv2.imwrite(f"{debug_prefix}_closed.png", closed)

    # 바깥 윤곽만 찾기
    contours, _ = cv2.findContours(
        closed.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise RuntimeError("컨투어를 하나도 못 찾았음")

    img_area = image.shape[0] * image.shape[1]
    min_area_ratio = 0.2   # 전체 이미지의 20% 이상인 것만 문서 후보로 사용

    document_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * min_area_ratio:
            # 너무 작은 건 무시
            continue

        # 가장 큰 컨투어를 문서로 사용
        if area > max_area:
            max_area = area
            document_contour = c

    if document_contour is None:
        # 조건에 안 걸리면 그냥 최대 컨투어 사용
        document_contour = max(contours, cv2.contourArea)

    # 가능하면 4개 꼭짓점으로 근사
    # 안 되면 회전 사각형 사용
    peri = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(document_contour)
        box = cv2.boxPoints(rect)
        quad = box.astype(np.float32)

    if debug_prefix:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [quad.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_prefix}_contour.png", debug_img)

    # 리사이즈 전에 좌표계로 되돌리기
    doc_pts = quad * ratio

    # 퍼스펙티브 변환으로 문서 전체 펴기
    warped = four_point_transform(orig, doc_pts)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_warped.png", warped)

    return warped


def process_document_image(img_path, debug_prefix="doc"):
    """
    단일 문서 사진 처리 전체 파이프라인

    1. 문서 윤곽 검출 및 퍼스펙티브 보정
    2. 결과 이미지 저장
    """
    warped = detect_document_and_warp(
        img_path,
        debug_prefix=debug_prefix
    )

    out_path = f"{debug_prefix}_warped.png"
    cv2.imwrite(out_path, warped)
    print(f"퍼스펙티브 보정된 문서 저장: {out_path}")

    return warped


if __name__ == "__main__":
    # 테스트용 이미지 경로
    img_path = "../exam_doc.jpg"   # 네가 가진 단일 문서 사진 파일 이름에 맞게 수정

    warped = process_document_image(
        img_path,
        debug_prefix="doc"
    )

    # 화면으로 확인하고 싶으면 아래 주석 해제
    # cv2.imshow("Warped Document", warped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
