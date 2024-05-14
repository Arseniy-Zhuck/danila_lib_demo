import cv2

from danila.danila import Danila

img_path = 'demo/WhatsApp Image 2023-05-23 at 07.34.17 (1).jpeg'
img = cv2.imread(img_path)
danila = Danila(3,'yolov5')
res = danila.rama_classify(img)
cv2.namedWindow(res, cv2.WINDOW_AUTOSIZE)
cv2.imshow(res, img)
cv2.waitKey(0)
new_img = danila.rama_detect(img)
cv2.namedWindow('rama', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rama', new_img)
cv2.waitKey(0)
new_img = danila.rama_cut(img)
cv2.namedWindow('rama', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rama', new_img)
cv2.waitKey(0)
new_img = danila.text_detect_cut(img)
cv2.namedWindow('rama', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rama', new_img)
cv2.waitKey(0)
new_img = danila.text_detect(img)
cv2.namedWindow('rama', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rama', new_img)
cv2.waitKey(0)
labels = danila.text_recognize(img)
print(labels)

