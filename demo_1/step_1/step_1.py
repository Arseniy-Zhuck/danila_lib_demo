import cv2

from danila import Danila

img_path = 'demo/step_1/example.jpeg'
img = cv2.imread(img_path)
danila = Danila()
res = danila.rama_classify(img_path)

cv2.namedWindow(res, cv2.WINDOW_AUTOSIZE)
cv2.imshow(res, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
