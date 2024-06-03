import json

import cv2

from danila.danila import Danila

img_path = 'vagon_text_recognize/0be3569c-54607833.jpg'
img = cv2.imread(img_path)
danila = Danila(4, 'yolov5')
result = danila.vagon_number_recognize(img, 512)
x = json.dumps(result, default=lambda  x: x.__dict__)
print(x)
# new_img = danila.vagon_number_detect(img, 512)
# cv2.namedWindow('vagon', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('vagon', new_img)
# img_path2 = 'vagon_text_recognize/29163987.jpg'
# img2 = cv2.imread(img_path2)
# new_img2 = danila.vagon_number_detect(img2, 512)
# cv2.namedWindow('vagon2', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('vagon2', new_img2)
# img_path3 = 'vagon_text_detect/29310091.jpg'
# img3 = cv2.imread(img_path3)
# new_img3 = danila.vagon_number_detect(img3, 512)
# cv2.namedWindow('vagon3', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('vagon3', new_img3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
