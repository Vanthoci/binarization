import cv2 as cv
import numpy as np

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
    cv.imwrite("result.jpg", res)
    cv.imshow("result", res)
    cv.waitKey(0)
    print("DONE")



