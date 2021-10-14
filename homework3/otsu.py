#coding:utf-8
import cv2
from matplotlib import pyplot as plt
import time

image = cv2.imread("./source.jpg", cv2.IMREAD_COLOR)
# 变换图像通道
b, g, r = cv2.split(image)
image = cv2.merge([r, g, b])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plt.subplot(121), plt.imshow(image), plt.title('source')
# ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
# plt.subplot(122), plt.imshow(th1, "gray")
# plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(131), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.hist(image.ravel(), 256)
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
plt.subplot(133), plt.imshow(th1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
plt.show()
