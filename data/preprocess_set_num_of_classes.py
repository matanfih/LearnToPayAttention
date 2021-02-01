import random
import os
import pandas as pd
import glob
import shutil

NEEDED_NUM_OF_CLASSES = 5
FILE_NAME = "Data_Entry_2017_v2020"
EXT = "csv"
CSV_FILE_NAME = "{}.{}".format(FILE_NAME, EXT)

NEW_SUFFIX = "_Top{}".format(NEEDED_NUM_OF_CLASSES)

print('Im here %s' % os.getcwdb())

complete_csv_path = os.path.join(os.getcwd(), CSV_FILE_NAME)

if not os.path.exists(complete_csv_path):
    print('hell no, did not find %s' % complete_csv_path)
    exit(1)


xray_frame = pd.read_csv(complete_csv_path)

image_list = xray_frame["Image Index"].tolist()
labels = xray_frame["Finding Labels"].tolist()

print("finished reading %s, found %s images" % (CSV_FILE_NAME, len(labels)))

print("explore data")
#multi_label_image = next(l for l in labels if '|' in l)
#print(multi_label_image.split('|'))
multi_labels = [(l, i) for l, i in zip(labels, image_list) if '|' in l]
print("amount of multi label images: %s" % len(multi_labels))

all_classes = set()
for l in labels:
    for p in l.split('|'):
        all_classes.add(p)

print("all available classes[%s]: %s" % (len(all_classes), all_classes))
label_map = {c: {'unique': [], 'multi-label': [], 'all': []} for c in all_classes}

for lbl, img in zip(labels, image_list):
    if lbl in label_map:
        label_map[lbl]['unique'].append(img)
    else:
        for l in lbl.split('|'):
            label_map[l]['multi-label'].append(img)
    #label_map[lbl]['all'].append(img)

for lbl in label_map.keys():
    label_map[lbl]['all'] = label_map[lbl]['unique'] + label_map[lbl]['multi-label']

print({lbl: {'unique': len(label_map[lbl]['unique']), 'multi-label': len(label_map[lbl]['multi-label'])} for lbl in label_map.keys()})
import operator
label_summary = sorted({lbl: (len(label_map[lbl]['unique']) + len(label_map[lbl]['multi-label'])) for lbl in label_map.keys()}.items(), key=operator.itemgetter(1), reverse=True)

top_needed_classes = []

#('No Finding', 60361), ('Infiltration', 19894), ('Effusion', 13317), ('Atelectasis', 11559)
NO_FINDING = 'No Finding'
need = NEEDED_NUM_OF_CLASSES

for lbl in label_summary:
    if need == 0:
        break

    if NO_FINDING in lbl:
        continue

    top_needed_classes.append(lbl)
    need -= 1

print("top %s summary[unique + multi]: %s" % (NEEDED_NUM_OF_CLASSES, [lbl for lbl in label_summary if lbl[0] in [t[0] for t in top_needed_classes]]))


def take_2nd(e):
    return e[1]


top_needed_classes_image_list = set()

for _class in sorted(top_needed_classes, reverse=False, key=take_2nd):
    before = len(top_needed_classes_image_list)
    for l in label_map[_class[0]]['unique'] + label_map[_class[0]]['multi-label']:
        top_needed_classes_image_list.add(l)

    print("collecting data from %s added %s out of %s available" % (_class[0], len(top_needed_classes_image_list) - before, _class[1]))

print("average for each class: %s" % int(len(top_needed_classes_image_list) * 1.0 / len(top_needed_classes)))
for img in (label_map[NO_FINDING]["unique"] + label_map[NO_FINDING]["multi-label"])[:int(len(top_needed_classes_image_list) * 1.0 / len(top_needed_classes))]:
    top_needed_classes_image_list.add(img)

top_xray_frames = xray_frame[xray_frame['Image Index'].isin(list(top_needed_classes_image_list))]
top_xray_frames_clone = top_xray_frames.copy(deep=True)
print(top_xray_frames.shape)

allowed_labels = [l[0] for l in top_needed_classes]
for i, img in enumerate(top_xray_frames_clone['Image Index']):
    print("handling %s:%s out of %s" % (i, img, len(top_xray_frames_clone['Image Index'])))
    expected_label = next((lbl for lbl in label_map.keys() if img in label_map[lbl]['all']), NO_FINDING)
    index = top_xray_frames_clone.index[top_xray_frames_clone['Image Index'] == img]
    current_label = top_xray_frames.loc[index, 'Finding Labels']
    current_label = current_label.values[0].split('|')
    allowed_subset_current = set(current_label) - (set(current_label) - set(allowed_labels))
    if len(current_label) == 1 or len(allowed_subset_current) == 1:
        if NO_FINDING in current_label:
            assert len(allowed_subset_current) == 0
            #top_xray_frames.loc[index, 'Finding Labels'] = NO_FINDING
            #print("curr: %s, skipping" % current_label)
        else:
            set_val = list(allowed_subset_current)[0]
            print("curr: %s, expected: %s, allowed: %s, subset: %s, set_val: %s" % (
                current_label, expected_label, allowed_labels, allowed_subset_current, set_val)
            )
            #assert set_val == current_label[0], "curr: {}, set_val: {} , allowed_subset_current: {}, skipping".format(current_label, set_val, allowed_subset_current)
            if set_val == current_label[0] and len(current_label) == 1:
                print("curr: %s, set_val: %s , skipping" % (current_label, set_val))
            else:
                print("curr: %s, set_val: %s , setting val" % (current_label, set_val))
                top_xray_frames.loc[index, 'Finding Labels'] = set_val
    else:
        assert len(allowed_subset_current) > 0, "curr:{} allow:{} expect:{}".format(current_label, allowed_subset_current, expected_label)
        set_val = '|'.join(allowed_subset_current)
        print("curr: %s, expected: %s, allowed: %s, subset: %s, set_val: %s" % (
            current_label, expected_label, allowed_labels, allowed_subset_current, set_val)
              )
        top_xray_frames.loc[index, 'Finding Labels'] = set_val
    #print("curr: %s, expected: %s, allowed: %s, subset: %s" % (current_label, expected_label, allowed_labels, allowed_subset_current))
    #top_xray_frames.loc[index, 'Finding Labels'] = label

top_xray_frames.to_csv('{}{}.{}'.format(FILE_NAME, NEW_SUFFIX, EXT), header=True, index=False)
