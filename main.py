import cv2, pdf2image
import numpy as np
from PIL import Image
from easyocr import Reader

shadow = False
resize = False
gray = False
denoise = False
brightness = False
threshold = False
deskew = False
crop = False
thicken = False
