import cv2, os
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

#################### pdf -> images converter ####################
def pdf2images(pdf, output_path):
    images = convert_from_path(pdf)
    for i, image in enumerate(images):
        image_path = f'{output_path + str(i)}.png'
        image.save(image_path, 'PNG')

#################### images -> pdf converter ####################
def images2pdf(imgs, output_pdf):
    # exts에 포함되는 확장자만 가져오기
    exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(imgs) if f.lower().endswith(exts)]
    files.sort()  # 이름 순 정렬

    if not files:
        print("이미지 파일이 없습니다.")
        return

    images = []
    for file in files:
        img_path = os.path.join(imgs, file)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    # 첫 이미지를 기준으로 순서대로 PDF 생성
    first = images[0]
    rest = images[1:]
    first.save(output_pdf, save_all=True, append_images=rest)


#################### shadow removal ####################
# Morphology 기반 그림자 제거 함수
# - 컬러 이미지를 LAB 색공간으로 변환
# - 밝기 채널(L)에서만 배경(조명)을 추정해서 그림자 성분을 줄임
# - 색상 정보(A, B 채널)는 최대한 그대로 유지해서 색감 보존

def removeshadow(path):
    # 1) 이미지 읽기 (BGR 형식)
    img = cv2.imread(path)

    # 2) BGR → LAB 변환
    #    L  채널: 밝기 정보
    #    A, B 채널: 색상 정보
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 3) 밝기 채널에서 배경(조명) 추정을 위한 커널 생성
    #    - 커널이 클수록 넓은 영역을 평균적으로 보게 되어
    #      전체적인 조명 패턴을 더 잘 잡을 수 있음
    kernel = np.ones((16 ,16), np.uint8)

    # 4) 팽창(dilation)으로 L 채널에서 밝은 영역을 확장
    #    - 어두운 그림자 부분은 채워지고
    #      전체적으로 부드러운 배경 밝기 맵을 만들기 위한 전처리
    dilated = cv2.dilate(L, kernel)

    # 5) medianBlur로 배경 밝기 맵을 더 부드럽게 만듦
    #    - 커널 크기(255)는 매우 강한 블러를 의미
    #    - 너무 작으면 세부 명암까지 따라가서 그림자 제거 효과가 약해짐
    bg = cv2.medianBlur(dilated, 255)

    # 6) 그림자 보정
    #    - absdiff(L, bg)  로 L과 배경 밝기 차이를 구한 뒤
    #    - 255 - 그 값을 해서 밝기를 반전시키는 효과를 줌
    #    - 어두운 그림자 부분은 상대적으로 더 밝게 보정됨
    diff = 255 - cv2.absdiff(L, bg)

    # 7) 보정된 L 채널을 0~255 범위로 다시 정규화
    #    - 연산 과정에서 대비가 너무 낮아지거나 범위가 줄어드는 것을 방지
    norm_L = cv2.normalize(
        diff,
        None,            # 출력(새 배열 생성)
        alpha=0,         # 최소값
        beta=255,        # 최대값
        norm_type=cv2.NORM_MINMAX  # 최소·최대 기준으로 스케일링
    )

    # 8) 색상 정보 A, B 채널은 그대로 두고
    #    밝기 채널만 보정된 L 채널로 교체
    merged_lab = cv2.merge([norm_L, A, B])

    # 9) 다시 LAB → BGR로 변환해서 최종 컬러 이미지로 반환
    result = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return result


def main():
    # 원하는 작업만 선택해서 True로 변환
    img2pdf = False
    pdf2img = False
    shadow = True
    gray = True
    adaptive = True

    # 경로(임시로 지정)
    img_path = "./imgs/ocr_test7.jpg"

    # 확장자 추출(jpg, png, jpeg 등 이미지의 확장자를 추출)
    ext = img_path.split(".")[-1]

    pdf_path = "./pdfs/conv_test.pdf"
    save_imgs_path = "./pdfs/img"

    imgs_path = "./pdfs"
    save_pdf_path = "./pdfs/merged_pdf.pdf"

    if img2pdf:
        images2pdf(imgs_path, save_pdf_path)

    if pdf2img:
        images2pdf(pdf_path, save_pdf_path)

    if shadow:
        out = removeshadow(img_path)
        cv2.imwrite(f"output_shadow.{ext}", out)

    if gray:
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"output_gray.{ext}", gray_img)

    if adaptive:
        img = cv2.imread(img_path, 0)

        # 적응형 이진화
        # maxValue: 임계값을 넘을 때 줄 값 (보통 255)
        # adaptiveMethod:
        #   cv2.ADAPTIVE_THRESH_MEAN_C    주변 평균 사용
        #   cv2.ADAPTIVE_THRESH_GAUSSIAN_C 주변 가중 평균(가우시안) 사용
        # thresholdType: cv2.THRESH_BINARY             (픽셀값 > 임계값 이면 maxValue, 아니면 0)
        # blockSize: 임계값 계산할 지역 크기(홀수)
        # C: 계산된 평균에서 얼마나 빼줄지 (값이 클수록 더 어두운 픽셀만 흰색으로)
        out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 40)
        cv2.imwrite(f"output_adaptive.{ext}", out)

if __name__ == "__main__":
    main()