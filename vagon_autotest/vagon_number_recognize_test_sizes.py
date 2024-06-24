import json

import torch
import cv2
import os

from danila.danila import Danila
from data.result.word_compare_result import Word_compare_result

# from vagon_number_recognize.Vagon_number_recognize_class import Vagon_number_recognize_class
# from vagon_number_recognize.Vagon_number_recognize_class_v2 import Vagon_number_recognize_class_v2
# from vagon_number_recognize.Vagon_number_recognize_class_v3 import Vagon_number_recognize_class_v3


# model and dataset
dir_path = 'vagon_autotest_dataset/'
label_path = dir_path + 'labels.txt'
image_dir_path = dir_path + 'images'
danila = Danila(10, 'yolov5')


# useful addresses
image_dir = os.listdir(image_dir_path)

def compare(result_text, label_text):
    if result_text == '':
        return Word_compare_result.none
    elif result_text == label_text:
        return Word_compare_result.equal
    else:
        count_eq = 0
        index = 0
        while (index < len(label_text)) and (index < len(result_text)):
            if result_text[index] == label_text[index]:
                count_eq = count_eq + 1
            index += 1
        per_cent = float(count_eq) / len(label_text)
        if per_cent > 0.5:
            return Word_compare_result.partial
        else:
            return Word_compare_result.wrong
data = {}
with open(label_path) as f:
        for line in f:
            lst = line.split()
            data[lst[0]] = lst[1]
new_new_lines = []
for size_h in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]:
    for size_w in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]:
        test_results = 'vagon_autotest/tests_results/217_vagons_det_incr_9500_im' + '_' +str(size_h) + '_' + str(size_w) + '.txt'
        counts = {Word_compare_result.equal:0,Word_compare_result.partial:0,Word_compare_result.none:0,Word_compare_result.wrong:0}
        per_cents = {Word_compare_result.equal:0.0,Word_compare_result.partial:0.0,Word_compare_result.none:0.0,Word_compare_result.wrong:0.0}
        new_lines = []
        print(test_results)
        new_new_lines.append(test_results)
        new_new_lines.append('\n')
        n = 0
        for image_name in image_dir:
            n += 1
            img_path = image_dir_path + '/' + image_name
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            result = danila.vagon_number_recognize(img, 512, size_number_h=size_h, size_number_w=size_w)
            if result.number is None:
                result_text = ''
            else:
                result_text = result.number.text
            label_text = data[image_name]
            # if len(img_letters.letters) == len(label_letters.letters):
            #     flag = True
            #     for i in range(0, len(label_letters.letters)):
            #         flag = flag and (img_letters.letters[i].letter == label_letters.letters[i].letter)
            #     print(flag)
            # else:
            #     print(False)
            compare_result = compare(result_text,label_text)
            if compare_result != Word_compare_result.equal:
                print(str(n) + '. ' + image_name)
                print(result_text + ' - ' + label_text)
                new_lines.append(str(n) + '. ' + image_name)
                new_lines.append('\n')
                new_lines.append(result_text + ' - ' + label_text)
                new_lines.append('\n')
            counts[compare_result] += 1
        print(counts)
        new_lines.append(str(counts))
        new_new_lines.append(str(counts))
        new_new_lines.append('\n')
        for (result,count) in counts.items():
            per_cents[result] = round(count / float(len(image_dir)), 3) * 100
        print(per_cents)
        new_lines.append('\n')
        new_lines.append(str(per_cents))
        new_new_lines.append(str(per_cents))
        new_new_lines.append('\n')
        with open(test_results, "w") as new_f:
            new_f.writelines(new_lines)
test_results = 'vagon_autotest/tests_results/217_vagons_det_incr_9500_im_all_sizes.txt'
with open(test_results, "w") as new_f:
    new_f.writelines(new_new_lines)

