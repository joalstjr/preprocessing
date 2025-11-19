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
    책 양면 전체를 하나의 직사각형으로 펴는 용도
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


def detect_double_page_and_warp(img_path, debug_prefix=None):
    """
    책 양면 사진에서
    1. 책 전체 윤곽(사각형) 검출
    2. 퍼스펙티브 보정으로 양면 평평하게 펴기

    Canny 대신 Otsu 이진화 기반으로 책 전체를 검출함
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

    # Otsu 이진화로 책 영역을 덩어리로 만들기
    # 배경(노란 담요, 바닥)보다 페이지가 더 밝으니까
    # THRESH_BINARY 사용
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
            # 너무 작은 건 (05 박스처럼) 무시
            continue

        # 가장 큰 컨투어를 문서로 사용
        if area > max_area:
            max_area = area
            document_contour = c

    if document_contour is None:
        # 조건에 안 걸리면 그냥 최대 컨투어 사용
        document_contour = max(contours, key=cv2.contourArea)

    # 가능하면 4개 꼭짓점으로 근사해보고,
    # 안 되면 minAreaRect 써서 사각형 만들어줌
    peri = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        # 회전 사각형으로 보정
        rect = cv2.minAreaRect(document_contour)
        box = cv2.boxPoints(rect)
        quad = box.astype(np.float32)

    if debug_prefix:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [quad.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_prefix}_contour.png", debug_img)

    # 리사이즈 전에 좌표계로 되돌리기
    doc_pts = quad * ratio

    # 퍼스펙티브 변환으로 양면 전체 펴기
    warped = four_point_transform(orig, doc_pts)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_warped.png", warped)

    return warped


def split_double_page(warped, gutter_margin=10, search_region_ratio=0.3,
                      debug_prefix=None):
    """
    펴진 양면 이미지에서 가운데 책등 위치를 찾아
    왼쪽 페이지, 오른쪽 페이지로 분리

    gutter_margin: 책등 기준으로 양쪽 몇 픽셀 잘라낼지
    search_region_ratio: 가운데 몇 퍼센트 구간만 후보로 볼지 (0.3 이면 중앙 30%)
    """
    h, w = warped.shape[:2]

    # 그레이 변환
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 세로 방향으로 각 열의 평균 밝기 계산
    column_means = gray.mean(axis=0)

    # 1D 블러로 노이즈 제거
    kernel_size = 51
    kernel = np.ones(kernel_size) / kernel_size
    smooth_means = np.convolve(column_means, kernel, mode="same")

    # 중앙 근처만 후보로 사용
    center = w // 2
    half_range = int(w * search_region_ratio / 2)

    left_bound = max(0, center - half_range)
    right_bound = min(w - 1, center + half_range)

    local_region = smooth_means[left_bound:right_bound]
    local_min_index = int(np.argmin(local_region))
    gutter_x = left_bound + local_min_index

    # 디버깅용: 책등 위치 표시
    if debug_prefix:
        debug = warped.copy()
        cv2.line(debug, (gutter_x, 0), (gutter_x, h), (0, 0, 255), 2)
        cv2.imwrite(f"{debug_prefix}_gutter.png", debug)

        # 평균 밝기 그래프를 이미지처럼 보고 싶으면 아래처럼 간단히 시각화
        graph_h = 200
        graph = np.full((graph_h, w, 3), 255, dtype=np.uint8)
        # 정규화해서 0~graph_h 범위로
        m_min, m_max = smooth_means.min(), smooth_means.max()
        denom = (m_max - m_min) if (m_max - m_min) != 0 else 1
        norm = (smooth_means - m_min) / denom
        ys = (graph_h - 1 - norm * (graph_h - 1)).astype(int)
        for x in range(w):
            cv2.circle(graph, (x, ys[x]), 1, (0, 0, 0), -1)
        cv2.line(graph, (gutter_x, 0), (gutter_x, graph_h - 1), (0, 0, 255), 1)
        cv2.imwrite(f"{debug_prefix}_column_mean.png", graph)

    # 책등 주변 margin 만큼 잘라서 좌우 페이지 분리
    left_end = max(0, gutter_x - gutter_margin)
    right_start = min(w, gutter_x + gutter_margin)

    left_page = warped[:, :left_end]
    right_page = warped[:, right_start:]

    return left_page, right_page


def process_book_image(img_path,
                       debug_prefix="book_debug",
                       gutter_margin=10,
                       search_region_ratio=0.3):
    """
    책 양면 사진 처리 전체 파이프라인

    1. 양면 문서 윤곽 검출 및 퍼스펙티브 보정
    2. 책등 자동 검출
    3. 왼쪽 / 오른쪽 페이지 잘라서 각각 저장
    """
    # 1단계: 양면 전체 펴기
    warped = detect_double_page_and_warp(
        img_path,
        debug_prefix=debug_prefix
    )

    # 2단계: 책등 기준으로 좌우 페이지 분리
    left_page, right_page = split_double_page(
        warped,
        gutter_margin=gutter_margin,
        search_region_ratio=search_region_ratio,
        debug_prefix=debug_prefix
    )

    # 결과 저장
    left_path = f"{debug_prefix}_left.png"
    right_path = f"{debug_prefix}_right.png"
    cv2.imwrite(left_path, left_page)
    cv2.imwrite(right_path, right_page)

    print(f"왼쪽 페이지 저장: {left_path}")
    print(f"오른쪽 페이지 저장: {right_path}")

    return left_page, right_page


if __name__ == "__main__":
    # 테스트용 이미지 경로
    img_path = "shadow_test1.png"   # 네가 올린 사진 파일 이름에 맞게 수정

    # 기본 파라미터로 실행
    left, right = process_book_image(
        img_path,
        debug_prefix="book",
        gutter_margin=12,
        search_region_ratio=0.1  # 여기만 수정!
    )

    # 화면으로도 확인하고 싶으면 아래 주석 해제
    # cv2.imshow("Left Page", left)
    # cv2.imshow("Right Page", right)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
