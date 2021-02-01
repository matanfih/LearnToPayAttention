from __future__ import print_function, division

import random
import shutil
from glob import glob
import csv, time
import PIL
import torch
import json

import torchvision
from PIL.Image import Image

import os
import pathlib

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split


def resize128_to_tensor():
    return transforms.Compose([transforms.Resize(128, PIL.Image.BICUBIC), transforms.ToTensor()])


def to_PIL_resize1024():
    return transforms.Compose([transforms.ToPILImage(), transforms.Resize(1024, PIL.Image.BICUBIC)])


class XrayDataset(Dataset):
    CSV_FIXED_PATH_FILE_NAME_PREFIX = "fixed_path_"
    TRAIN = "train_"
    TEST = "test_"

    LABEL_TAB = "Finding Labels"
    IMAGE_TAB = "Image Index"

    FIXED_TRAIN = TRAIN + CSV_FIXED_PATH_FILE_NAME_PREFIX
    FIXED_TEST = TEST + CSV_FIXED_PATH_FILE_NAME_PREFIX

    def __pre_process(self, single_label=None):
        images = glob("{}/images_*/*.png".format(self.root_dir))
        not_found = "None"
        if single_label is not None:
            image_list = []
            labels = []
            single_label_len = len([l for l in self.xray_frame[self.LABEL_TAB].tolist() if single_label in l])
            for img, lbl in zip(self.xray_frame[self.IMAGE_TAB].tolist(), self.xray_frame[self.LABEL_TAB].tolist()):
                if single_label in lbl:
                    image_list.append(img)
                    labels.append(lbl)
                elif single_label_len > 0:
                    image_list.append(img)
                    labels.append(not_found)
                    single_label_len -= 1

        else:
            image_list = self.xray_frame[self.IMAGE_TAB].tolist()
            labels = self.xray_frame[self.LABEL_TAB].tolist()

        self_len = self.__len__()
        index = 1
        for img in images:
            name = os.path.basename(img)
            index += 1
            if name not in image_list:
                continue
            i = image_list.index(name)
            image_list[i] = img
            print("images left: %s" % (len(images) - index))

        self.class_to_idx = {}
        for l in labels:
            if single_label is not None:
                if single_label in self.class_to_idx and not_found in self.class_to_idx:
                    break
                elif not_found in l:
                    self.class_to_idx[not_found] = 0
                elif single_label in l:
                    self.class_to_idx[single_label] = 1

            else:
                for pathology in l.split('|'):
                    if pathology not in self.class_to_idx:
                        self.class_to_idx[pathology] = 0
                    self.class_to_idx[pathology] += 1

        print("labels distribution: %s" % self.class_to_idx)

        index = 1
        # if single_label:
        #     self.class_to_idx = {single_label: 1}
        # else:
        if single_label is None:
            for k in self.class_to_idx.keys():
                if "No Finding" in k:
                    self.class_to_idx[k] = 0
                else:
                    self.class_to_idx[k] = index
                    index += 1

        print("label translation: %s" % self.class_to_idx)

        with open(self.meta_json_path, 'w') as fp:
            json.dump(self.class_to_idx, fp=fp, indent=4)

        print("image list len: %s, labels len %s" % (len(image_list), len(labels)))
        X_train, X_test, y_train, y_test = train_test_split(image_list, labels, test_size=0.2, random_state=42)

        for X, y, filename in [(X_train, y_train, self.train_csv_fixed_path),
                               (X_test, y_test, self.test_csv_fixed_path)]:
            print("writing the X: %s, y: %s -> %s" % (len(X), len(y), filename))
            with open(filename, 'w') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["Image Index", "Finding Labels"])
                writer.writerows(zip(X, y))
        print("%s finished" % self.__pre_process.__name__)

    def __init__(self, csv_file, root_dir, transform=None, train=True, force_pre_process=False,
                 single_label="Infiltration"):

        if not os.path.exists(root_dir):
            raise Exception("ohh snap!! nih directory not found [%s]" % root_dir)

        #self.orig_csv_path = os.path.join(root_dir, csv_file)
        self.orig_csv_path = csv_file
        csv_file_name, csv_dir = os.path.basename(self.orig_csv_path), os.path.dirname(self.orig_csv_path)

        self.single_label = single_label
        if single_label is None:
            self.test_csv_fixed_path = os.path.join(csv_dir, "{}{}".format(self.FIXED_TEST, csv_file_name))
            self.train_csv_fixed_path = os.path.join(csv_dir, "{}{}".format(self.FIXED_TRAIN, csv_file_name))
            self.meta_json_path = os.path.join(csv_dir, "meta.json")
        else:
            self.test_csv_fixed_path = os.path.join(csv_dir, "{}_{}{}".format(single_label, self.FIXED_TEST, csv_file_name))
            self.train_csv_fixed_path = os.path.join(csv_dir, "{}_{}{}".format(single_label, self.FIXED_TRAIN, csv_file_name))
            self.meta_json_path = os.path.join(csv_dir, "{}_meta.json".format(single_label, single_label))
        self.root_dir = root_dir

        e = [d for d in (self.test_csv_fixed_path, self.train_csv_fixed_path, self.meta_json_path) if not os.path.exists(d)]
        if len(e) != 0 or force_pre_process:
            self.xray_frame = pd.read_csv(self.orig_csv_path)
            self.__pre_process(single_label=single_label)

        if train:
            self.xray_frame = pd.read_csv(self.train_csv_fixed_path)
        else:  # test
            self.xray_frame = pd.read_csv(self.test_csv_fixed_path)

        self.transform = transform

        self._load_meta()
        self._validate_labels()

    def _validate_labels(self):
        for image, labels in self.xray_frame.iterrows():
            target = self.str_to_target(labels[1])

    def _load_meta(self):
        path = self.meta_json_path
        with open(path, 'rb') as infile:
            data = json.load(infile)
        self.class_to_idx = data
        self.idx_to_class = {i: _class for _class, i in self.class_to_idx.items()}

    def str_to_target(self, s_target):
        if self.single_label is not None:
            if self.single_label in s_target:
                return torch.tensor(1, dtype=torch.long)
            else:
                return torch.tensor(0, dtype=torch.long)
        else:
            if '|' in s_target:
                #print("found multi label image [%s], taking first" % s_target)
                s_target = s_target.split('|')
                #assert len(s_target) > 3, s_target
                labels = [self.class_to_idx[s_t] for s_t in s_target]
            else:
                labels = self.class_to_idx[s_target]

            hot_labels = [0] * (len(self.class_to_idx) - 1)
            if "No Finding" not in s_target:
                try:
                    for l in labels:
                        hot_labels[int(l) -1] = 1
                except:
                    hot_labels[int(labels) - 1] = 1

            return torch.tensor(hot_labels, dtype=torch.float)
            #return torch.tensor(self.class_to_idx[s_target], dtype=torch.long)

    def __len__(self):
        return len(self.xray_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.xray_frame.iloc[idx]["Image Index"]
        image = PIL.Image.open(img_name).convert('L')
        xray = self.xray_frame.iloc[idx, 1]
        #print("get image:%s, target:%s" % (image.size, xray))
        xray = self.str_to_target(xray)

        if self.transform:
            image = self.transform(image)

        return image.float(), xray


class PacemakerDataset(Dataset):
    def __init__(self, transform, root='/data/matan/nih', csv_file='data/pacemakers-nih-export.csv', is_train=True, only_train=True):
        self._xray_map = {"train": {"image": [], "label": []},
                            "test" :{"image": [], "label": []}}
        self.transform = transform
        self._root = root
        self.meta = os.path.join('data', 'meta')
        self.pacmaker_frames = pd.read_csv(csv_file)
        self._is_train = is_train
        self._only_train = only_train

        self.pacemaker_labels = self.pacmaker_frames["label"].tolist()
        self.pacemaker_image_names = self.pacmaker_frames["image"].tolist()

        self.pace_pairs = {img: lbl for img, lbl in zip(self.pacemaker_image_names, self.pacemaker_labels)}

        self.pacemaker_types = set(self.pacemaker_labels)
        self.pacemaker_types.add('no_finding')

        self.torchvision_path = \
            os.path.join(self._root, 'chest_xray_pacemaker_{}'.format(len(self.pacemaker_image_names)))

        if not os.path.exists(self.torchvision_path):
            self.pre_process()

        self.load_meta()

    def load_meta(self):
        with open(self.meta, 'r') as mj:
            self._xray_map = json.load(mj)
        self.class_to_idx = self._xray_map["class_to_idx"]
        self.train_xray_frame = pd.DataFrame(self._xray_map["train"])
        self.test_xray_frame = pd.DataFrame(self._xray_map["test"])

    def class_to_index(self, _class=None):
        if _class is None:
            return self.class_to_idx
        else:
            return self.class_to_idx[_class]

    def __len__(self):
        if self._is_train:
            return len(self.train_xray_frame["image"])
        else:
            return len(self.test_xray_frame["image"])

    def __getitem__(self, idx):
        frame = self.train_xray_frame if self._is_train else self.test_xray_frame
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = frame.iloc[idx]["image"]
        image = PIL.Image.open(img_name).convert('L')
        label = frame.iloc[idx, 1]
        #print("get image:%s, target:%s" % (image.size, xray))
        #xray = self.str_to_target(xray)

        if self.transform:
            image = self.transform(image)

        return image.float(), label

    def pre_process(self):
        paths = []
        __base_path = os.path.join(self._root, 'chest_xray_pacemaker_{}'.format(len(self.pacemaker_image_names)))
        for _t in self.pacemaker_types:
            __lbl = os.path.join(__base_path, _t)
            paths.append(os.path.join(__lbl, 'train'))
            paths.append(os.path.join(__lbl, 'test'))

        for p in paths:
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)

        __images = glob("{}/images_*/*.png".format(self._root))
        _no_finding = _finding = len(self.pacemaker_image_names)

        pm_real_path = [img for img in __images if os.path.basename(img) in self.pacemaker_image_names]

        print("pace unique: ", len(set(self.pacemaker_image_names)))
        missing = set(self.pacemaker_image_names) - set([os.path.basename(pm) for pm in pm_real_path])
        print("pace missing: ", len(missing))

        assert len(missing) == 0

        for _pm in pm_real_path:
            _base = os.path.basename(_pm)
            _path = os.path.join(__base_path, self.pace_pairs[_base])
            if not self._only_train:
                raise NotImplemented()
            shutil.copy(_pm, os.path.join(_path, 'train', _base))

        _rnd_no_find = random.sample(range(len(__images)), len(self.pacemaker_image_names) * 2)
        _no_finding_needed = len(self.pacemaker_image_names)
        for _irnd in _rnd_no_find:
            _no_pm = __images[_irnd]
            _name = os.path.basename(_no_pm)
            if _name in self.pace_pairs:
                continue
            _base = "no_finding"
            _path = os.path.join(__base_path, _base)
            if not self._only_train:
                raise NotImplemented()
            shutil.copy(_no_pm, os.path.join(_path, 'train', _name))

            if _no_finding_needed == 0:
                break
            _no_finding_needed -= 1

        self._xray_map["class_to_idx"] = {_class: i for i, _class in enumerate(self.pacemaker_types)}
        for _type in self.pacemaker_types:
            for iteration in ["train", "test"]:
                _path = os.path.join(self.torchvision_path, _type, iteration)
                for img in glob("{}/*.png".format(_path)):
                    self._xray_map[iteration]["image"].append(img)
                    self._xray_map[iteration]["label"].append(self._xray_map["class_to_idx"][_type])

        if os.path.exists(self.meta):
            os.remove(self.meta)

        with open(self.meta, 'w') as jmeta:
            json.dump(self._xray_map, jmeta, indent=4)


class XRAY(object):
    def __init__(self, train_image_transform, test_image_transform, force_pre_process=False,
                 csv_file=None, root_dir=None):
        csv_file = 'Data_Entry_2017_v2020.csv' if csv_file is None else csv_file
        root_dir = '/data/matan/nih' if root_dir is None else root_dir

        csv_path = csv_file if os.path.exists(csv_file) else os.path.join(root_dir, csv_file)

        assert os.path.exists(csv_path)

        self.train_set = XrayDataset(csv_file=csv_path, root_dir=root_dir, train=True,
                                     transform=train_image_transform, force_pre_process=force_pre_process,
                                     single_label=None)

        self.test_set = XrayDataset(csv_file=csv_path, root_dir=root_dir, train=False,
                                    transform=test_image_transform,
                                    single_label=None)

    def class_to_index(self, _class=None):
        if _class is None:
            return self.train_set.class_to_idx
        else:
            return self.train_set.class_to_idx[_class]
