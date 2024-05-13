import cv2

from danila.danila import Danila

img_path = 'demo/example_1.jpeg'
img = cv2.imread(img_path)
danila = Danila(2,'yolov5')
labels = danila.text_recognize(img)
print(labels)

