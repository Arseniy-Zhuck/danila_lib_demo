import cv2

from danila.danila import Danila

img_path = 'demo_4/example.jpeg'
img = cv2.imread(img_path)
danila = Danila(2, 'yolov5')
new_img = danila.text_detect_cut(img)
cv2.namedWindow('rama', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rama', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

