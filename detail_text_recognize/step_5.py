import json

import cv2
from danila.danila import Danila

danila = Danila(4, 'yolov5')
img_paths = ['detail_classify/5029.jpeg', 'detail_classify/28032449.jpg', 'detail_classify/29163987.jpg', 'detail_classify/example.jpeg']
for img_path in img_paths:
    img = cv2.imread(img_path)

    result = danila.detail_text_recognize(img, 512)
    x = json.dumps(result, default=lambda  x: x.__dict__)
    print(img_path)
    print(x)