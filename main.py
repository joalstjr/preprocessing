import cv2, os
from PIL import Image
from pdf2image import convert_from_path

# 원하는 작업만 선택해서 True로 변환
img2pdf = False
pdf2img = False

def pdf_to_images(pdf, output_path):
    images = convert_from_path(pdf)
    for i, image in enumerate(images):
        image_path = f'{output_path + str(i)}.png'
        image.save(image_path, 'PNG')

def images_to_pdf(imgs, output_pdf):
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

# imgs to pdf
imgs_path = "./pdfs"
save_pdf_path = "./pdfs/merged_pdf.pdf"

# pdf to imgs
pdf_path = "./pdfs/conv_test.pdf"
save_imgs_path = "./pdfs/img"

pdf_to_images(pdf_path, save_imgs_path)
images_to_pdf(imgs_path, save_pdf_path)