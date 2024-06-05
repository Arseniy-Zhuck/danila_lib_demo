import json

import cv2
from danila.danila import Danila

img_path = 'rama_classify/5029.jpeg'
img = cv2.imread(img_path)
danila = Danila(4, 'yolov5')
result = danila.rama_classify(img, 512)
x = json.dumps(result, default=lambda  x: x.__dict__)
print(x)