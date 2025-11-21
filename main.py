import cv2, pdf2image
import numpy as np
from pdf2image import convert_from_path

pdf_path = "../pdfs/conv_test.pdf"
output_path = "../pdfs"

images = convert_from_path(pdf_path)
for i, image in enumerate(images):
    image_path = f'{output_path}{i}.png'
    image.save(image_path, 'PNG')
    print(f'saved {image_path}')