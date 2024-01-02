import cv2 as cv
import numpy as np
import pytesseract
import re

def normalize(text):
    pattern = r'[^a-zA-Z,.\s\n]'
    text = re.sub(pattern, '', text)

    text = re.sub(r'(?<!\n)\n(?!\n)', '', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text

def retina_bina(img, iter_step = 5):
    retina = cv.bioinspired_Retina.create((img.shape[1], img.shape[0]))

    img2 = img[:]
    for i in range(iter_step):
        retina.run(img2)
        img2 = retina.getParvo()

    _, result = cv.threshold(img2, 0, 255, cv.THRESH_OTSU)

    return result


if __name__ == '__main__':
    img = cv.imread("sample.jpg", cv.IMREAD_GRAYSCALE)
    res = retina_bina(img)
    txt = normalize(pytesseract.image_to_string(res))
    file = open("ocrText.txt", 'w')
    file.write(txt)
    print("DONE")



