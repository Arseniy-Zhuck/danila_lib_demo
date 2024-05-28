import os
import json
import cv2
from danila.danila import Danila
from data.result.word_compare_result import Word_compare_result
sizes = [512]
prods = ['begickaya', 'ruzhimmash']
for prod in prods:
    for size in sizes:
        v = 4
        version = 'v180'
        dir_path = 'autotests_dataset/' + prod
        test_results = 'autotests/results/' + version + '_' + prod + '_' + str(size) + '.txt'
        str1 = 'danila_' + version + '_' + prod
        danila = Danila(4, 'yolov5')
        label_dir_path = dir_path + '/' + 'labels'
        image_dir_path = dir_path + '/' + 'images'
        image_dir = os.listdir(image_dir_path)
        image_dir_res_path = 'autotests/results/' + version + '_' + prod + '_' + str(size)
        os.makedirs(image_dir_res_path, exist_ok=True)

        n = 0
        for image_name in image_dir:
            print(str(n) + '. ' + image_name)
            n += 1
            img_path = image_dir_path + '/' + image_name
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            res = danila.text_detect_cut(img, 512)
            cv2.imwrite(image_dir_res_path + '/' + image_name, res)