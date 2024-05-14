import os
import json
import cv2
from danila.danila import Danila
from data.result.word_compare_result import Word_compare_result

dir_path = 'autotests_dataset/begickaya'
test_results = 'autotests/results/1_begickaya'
str1 = 'danila-v2-bejickaya'
danila = Danila(1,'yolov5')
label_dir_path = dir_path + '/' + 'labels'
image_dir_path = dir_path + '/' + 'images'
image_dir = os.listdir(image_dir_path)

def compare_str(str_r, str_l):
    if len(str_r) == 0:
        return Word_compare_result.none
    else:
        if str_r == str_l:
            return Word_compare_result.equal
        else:
            count = 0
            for sym in str_r:
                if str_l.find(sym) > -1:
                    count += 1
            per_cent = count / float(len(str_l))
            return Word_compare_result(1) if per_cent < 0.5 else Word_compare_result(2)

def compare(result, label):
    res = {'whole' : Word_compare_result.none, 'number' : Word_compare_result.none, 'prod' : Word_compare_result.none, 'year' : Word_compare_result.none}
    res['number'] = compare_str(result['number'], label['number'])
    res['prod'] = compare_str(result['prod'], label['prod'])
    res['year'] = compare_str(result['year'], label['year'])
    if (res['number'] == Word_compare_result.equal
        ) and (
            res['prod'] == Word_compare_result.equal
        ) and (
            res['year'] == Word_compare_result.equal
    ):
        res['whole'] = Word_compare_result.equal
    else:
        count = 0
        count += res['number'].value
        count += res['prod'].value
        count += res['year'].value
        if count > 5:
            res['whole'] = Word_compare_result.partial
        elif count > 2:
            res['whole'] = Word_compare_result.wrong
        else:
            res['whole'] = Word_compare_result.none
    return res

counts = {'whole' : None, 'number' : None, 'year' : None, 'prod' : None}
per_cents = {'whole' : None, 'number' : None, 'year' : None, 'prod' : None}
for key in counts.keys():
    counts[key] = {Word_compare_result.equal:0,Word_compare_result.partial:0,Word_compare_result.none:0,Word_compare_result.wrong:0}
for key in per_cents.keys():
    per_cents[key] = {Word_compare_result.equal:0.0,Word_compare_result.partial:0.0,Word_compare_result.none:0.0,Word_compare_result.wrong:0.0}

new_lines = []
new_lines.append(str1)
n = 0
for image_name in image_dir:
    n += 1
    img_path = image_dir_path + '/' + image_name
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    result = danila.text_recognize(img)
    label_name = image_name[0:image_name.rfind('.')]
    label_path = label_dir_path + '/' + label_name + '.json'
    with open(label_path) as l:
        label = json.load(l)
    res = compare(result, label)
    for key in res.keys():
        counts[key][res[key]] += 1
    new_lines.append(str(n) + '. ' + image_name + '\n')
    new_lines.append(str(result))
    new_lines.append(str(label))
    print(str(n) + '. ' + image_name + '\n')
    print(result)
    print(label)
for key in counts.keys():
    for key1 in counts[key].keys():
        per_cents[key][key1] = round(counts[key][key1] / float(n), 3) * 100
new_lines.append(str(counts) + '\n')
new_lines.append(str(per_cents) + '\n')
print(counts)
print(per_cents)
with open(test_results, "w") as new_f:
    new_f.writelines(new_lines)

