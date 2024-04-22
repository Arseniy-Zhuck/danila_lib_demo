import cv2
from danila.danila import Danila

img_path = 'demo_1/example.jpeg'
img = cv2.imread(img_path)
danila = Danila('yolov5')
res = danila.rama_classify(img)
cv2.namedWindow(res, cv2.WINDOW_AUTOSIZE)
cv2.imshow(res, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
