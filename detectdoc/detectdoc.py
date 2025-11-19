import cv2
import numpy as np

# 전역 변수
points = []          # 사용자가 클릭한 점들 저장
image = None         # 원본 이미지
clone = None         # 그리기용 복사본


def order_points(pts):
    """
    사각형 네 점을 [좌상, 우상, 우하, 좌하] 순서로 정렬
    pts: (4, 2) 형태의 numpy 배열
    """
    rect = np.zeros((4, 2), dtype="float32")

    # x + y 가 가장 작은 점이 좌상, 가장 큰 점이 우하
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    # x - y 가 가장 작은 점이 우상, 가장 큰 점이 좌하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    네 꼭짓점 좌표로 퍼스펙티브 변환 수행
    선택한 영역을 반듯한 직사각형으로 펴기
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


def mouse_callback(event, x, y, flags, param):
    """
    마우스 왼쪽 버튼 클릭할 때마다 점 하나씩 추가
    4점 찍으면 사각형을 화면에 그려줌
    """
    global points, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        # 점 추가
        points.append((x, y))

        # 점 그리기
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

        # 4점 다 찍었으면 선도 그려줌
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(clone, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("image", clone)


def main():
    global image, clone, points

    img_path = "../exam_doc.jpg"   # 처리할 이미지 경로
    image = cv2.imread(img_path)

    if image is None:
        print("이미지를 못 불러옴:", img_path)
        return

    clone = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", clone)

    # 마우스 콜백 등록
    cv2.setMouseCallback("image", mouse_callback)

    print("설명")
    print("1. 창에서 네 꼭짓점 순서 상관없이 4번 클릭")
    print("2. 4점 찍고 나면 w 를 누르면 퍼스펙티브 보정 수행")
    print("3. r 을 누르면 점 초기화")
    print("4. q 를 누르면 종료")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord("q"):
            break

        # 점 초기화
        if key == ord("r"):
            points = []
            clone = image.copy()
            cv2.imshow("image", clone)
            print("점 초기화함")

        # 퍼스펙티브 보정 실행
        if key == ord("w"):
            if len(points) != 4:
                print("점 4개를 먼저 찍어야 함. 현재 개수:", len(points))
                continue

            pts = np.array(points, dtype="float32")
            warped = four_point_transform(image, pts)

            cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
            cv2.imshow("warped", warped)

            out_path = "manual_warped.png"
            cv2.imwrite(out_path, warped)
            print("보정된 이미지 저장:", out_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
