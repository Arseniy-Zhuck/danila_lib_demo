import cv2

from danila.danila import Danila

img_path = 'demo/example.jpeg'
img = cv2.imread(img_path)
danila = Danila('yolov5')
labels = danila.text_recognize(img)
print(labels)

