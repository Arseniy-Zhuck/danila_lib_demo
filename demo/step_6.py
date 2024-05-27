import cv2

from danila.danila import Danila

img_path = 'demo/example.jpeg'
img = cv2.imread(img_path)
danila = Danila(4,'yolov5')
res = danila.rama_classify(img)
cv2.namedWindow(res, cv2.WINDOW_AUTOSIZE)

new_img_rama_detect = danila.rama_detect(img)
cv2.namedWindow('rama_detect', cv2.WINDOW_AUTOSIZE)

new_img_rama_cut = danila.rama_cut(img)
cv2.namedWindow('rama_cut', cv2.WINDOW_AUTOSIZE)
new_img_text_detect_cut = danila.text_detect_cut(img)
cv2.namedWindow('text_detect_cut', cv2.WINDOW_AUTOSIZE)
new_img_text_detect = danila.text_detect(img)
cv2.namedWindow('text_detect', cv2.WINDOW_AUTOSIZE)
cv2.imshow(res, img)
cv2.imshow('rama_detect', new_img_rama_detect)
cv2.imshow('rama_cut', new_img_rama_cut)
cv2.imshow('text_detect_cut', new_img_text_detect_cut)
cv2.imshow('text_detect', new_img_text_detect)
cv2.waitKey(0)
labels = danila.text_recognize(img)
print(labels)

