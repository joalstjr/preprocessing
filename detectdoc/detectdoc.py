import cv2
import numpy as np

# 전역 변수
points = []   # 클릭한 점 좌표 저장
image = None  # 원본 이미지
clone = None  # 표시용 이미지


def order_points(pts):
    """
    네 점을 [좌상, 우상, 우하, 좌하] 순으로 정렬
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    네 점을 이용한 퍼스펙티브 변환
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


def mouse_callback(event, x, y, flags, param):
    """
    마우스 왼쪽 클릭으로 네 점 선택
    """
    global points, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            clone,
            str(len(points)),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(clone, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("image", clone)


def main():
    global image, clone, points

    img_path = "../exam_doc.jpg"  # 입력 이미지 경로
    image = cv2.imread(img_path)

    if image is None:
        print("이미지를 불러올 수 없음:", img_path)
        return

    clone = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", clone)

    cv2.setMouseCallback("image", mouse_callback)

    print("사용 방법")
    print("1. 이미지에서 네 꼭짓점을 순서 상관없이 클릭")
    print("2. 4점 선택 후 w 키: 퍼스펙티브 보정")
    print("3. r 키: 점 초기화")
    print("4. q 키: 종료")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            points = []
            clone = image.copy()
            cv2.imshow("image", clone)
            print("점 초기화")

        if key == ord("w"):
            if len(points) != 4:
                print("점 4개 필요, 현재 개수:", len(points))
                continue

            pts = np.array(points, dtype="float32")
            warped = four_point_transform(image, pts)

            cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
            cv2.imshow("warped", warped)

            out_path = "manual_warped.png"
            cv2.imwrite(out_path, warped)
            print("저장:", out_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
