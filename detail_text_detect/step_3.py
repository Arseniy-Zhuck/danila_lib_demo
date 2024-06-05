import json

import cv2
from danila.danila import Danila

danila = Danila(4, 'yolov5')
img_paths = ['detail_classify/5029.jpeg', 'detail_classify/28032449.jpg', 'detail_classify/29163987.jpg', 'detail_classify/example.jpeg']
index= 0
results = []
for img_path in img_paths:
    img = cv2.imread(img_path)
    result = danila.detail_text_detect(img, 512)
    results.append(result)
    cv2.namedWindow(img_path, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(img_path, results[index])
    index += 1
cv2.waitKey(0)
cv2.destroyAllWindows()
