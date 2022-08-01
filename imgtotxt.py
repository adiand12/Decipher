import numpy as np
import os
import cv2
import pytesseract
import base64
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different



def img2txt(image):
    nparr = np.fromstring(image, np.uint8)
    #img = cv2.imread(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Convert image to black and white
    new_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

    text = pytesseract.image_to_string(new_image, config="--psm 3", lang="hin")
    return text

