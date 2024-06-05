import os

files = []
test_results = 'vagon_autotest_dataset/labels.txt'
whole_dir = 'vagon_autotest_dataset/images'
work_dir = os.listdir(whole_dir)
for image in work_dir:
    files.append(image + '\n')
with open(test_results, "w") as new_f:
    new_f.writelines(files)