import cv2
import numpy as np


def order_points(pts):
    """
    사각형 네 점을 [좌상, 우상, 우하, 좌하] 순서로 정렬
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    네 점 기반 퍼스펙티브 변환
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document_and_warp(img_path, debug_prefix=None):
    """
    문서 윤곽 검출 후 퍼스펙티브 보정
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 못 찾음: {img_path}")

    orig = image.copy()

    target_height = 700
    ratio = image.shape[0] / float(target_height)
    new_width = int(image.shape[1] / ratio)
    image = cv2.resize(image, (new_width, target_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_gray.png", gray)
        cv2.imwrite(f"{debug_prefix}_thresh.png", thresh)
        cv2.imwrite(f"{debug_prefix}_closed.png", closed)

    contours, _ = cv2.findContours(
        closed.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise RuntimeError("컨투어를 찾지 못함")

    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.2

    document_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area > max_area:
            max_area = area
            document_contour = c

    if document_contour is None:
        document_contour = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(document_contour)
        quad = cv2.boxPoints(rect).astype(np.float32)

    if debug_prefix:
        debug = image.copy()
        cv2.drawContours(debug, [quad.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_prefix}_contour.png", debug)

    doc_pts = quad * ratio
    warped = four_point_transform(orig, doc_pts)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_warped.png", warped)

    return warped


def process_document_image(img_path, debug_prefix="doc"):
    """
    문서 이미지 전체 처리
    """
    warped = detect_document_and_warp(img_path, debug_prefix)

    out_path = f"{debug_prefix}_warped.png"
    cv2.imwrite(out_path, warped)
    print(f"저장됨: {out_path}")

    return warped


if __name__ == "__main__":
    img_path = "../exam_doc.jpg"

    warped = process_document_image(
        img_path,
        debug_prefix="doc"
    )
