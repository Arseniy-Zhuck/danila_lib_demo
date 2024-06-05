import os
import json
import cv2
from danila.danila import Danila
from data.result.word_compare_result import Word_compare_result
version = 'v198'
dir_path = 'vagon_autotest_dataset'
danila = Danila(4, 'yolov5')
image_dir_path = dir_path + '/' + 'images'
image_dir = os.listdir(image_dir_path)
image_dir_res_path = 'vagon_autotest/tests_results/' + version
os.makedirs(image_dir_res_path, exist_ok=True)
n = 0
for image_name in image_dir:
    print(str(n) + '. ' + image_name)
    n += 1
    img_path = image_dir_path + '/' + image_name
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    res = danila.vagon_number_detect(img, 512)
    cv2.imwrite(image_dir_res_path + '/' + image_name, res)