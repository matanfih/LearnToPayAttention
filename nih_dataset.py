from __future__ import print_function, division
from glob import glob
import csv, time
import PIL
import torch
import json
from PIL.Image import Image

import os
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

    FIXED_TRAIN = TRAIN + CSV_FIXED_PATH_FILE_NAME_PREFIX
    FIXED_TEST = TEST + CSV_FIXED_PATH_FILE_NAME_PREFIX

    def __pre_process(self, single_label=None, correct_path=False):
        images = glob("{}/images_*/*.png".format(self.root_dir))

        if single_label is not None:
            image_list = []
            labels = []
            single_label_len = len([l for l in self.xray_frame["Finding Labels"].tolist() if single_label in l])
            for img, lbl in zip(self.xray_frame["Image Index"].tolist(), self.xray_frame["Finding Labels"].tolist()):
                if single_label in lbl:
                    image_list.append(img)
                    labels.append(lbl)
                elif single_label_len > 0:
                    image_list.append(img)
                    labels.append('None')
                    single_label_len -= 1

        else:
            image_list = self.xray_frame["Image Index"].tolist()
            labels = self.xray_frame["Finding Labels"].tolist()

        self_len = self.__len__()
        index = 1
        for img in images:
            name = os.path.basename(img)
            index += 1
            if name not in image_list:
                continue
            i = image_list.index(name)
            image_list[i] = img
            print("images left: %s" % (self_len - index))

        self.class_to_idx = {}
        for l in labels:
            for pathology in l.split('|'):
                if pathology not in self.class_to_idx:
                    self.class_to_idx[pathology] = 0
                self.class_to_idx[pathology] += 1

        print("labels distribution: %s" % self.class_to_idx)

        index = 1
        if single_label:
            self.class_to_idx = {single_label: 1}
        else:
            for k in self.class_to_idx.keys():
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

    def __init__(self, csv_file, root_dir, transform=None, train=True, force_pre_process=False, single_label="Infiltration"):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if not os.path.exists(root_dir):
            raise Exception("ohh snap!! root directory not found [%s]" % root_dir)

        self.orig_csv_path = os.path.join(root_dir, csv_file)
        self.single_label = single_label
        if single_label is None:
            self.test_csv_fixed_path = os.path.join(root_dir, "{}{}".format(self.FIXED_TEST, csv_file))
            self.train_csv_fixed_path = os.path.join(root_dir, "{}{}".format(self.FIXED_TRAIN, csv_file))
            self.meta_json_path = os.path.join(root_dir, "meta.json")
        else:
            self.test_csv_fixed_path = os.path.join(root_dir, "{}_{}{}".format(single_label, self.FIXED_TEST, csv_file))
            self.train_csv_fixed_path = os.path.join(root_dir, "{}_{}{}".format(single_label, self.FIXED_TRAIN, csv_file))
            self.meta_json_path = os.path.join(root_dir, "{}_meta.json".format(single_label, single_label))
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
        # if '|' in s_target:
        #
        #     #classes = [self.class_to_idx[s] for s in s_target.split('|')]
        #     return classes[0]
        #     #labels = [0] * len(self.class_to_idx.keys())
        #     #for c in classes:
        #     #    labels[c - 1] = 1
        #     #return torch.tensor(labels, dtype=torch.float32)
        # else:
        #     return self.class_to_idx[s_target]
        # if "Infiltration" in s_target:
        #     return torch.tensor(1, dtype=torch.float32)
        # else:
        #     return torch.tensor(0, dtype=torch.float32)
        if self.single_label is not None:
            if self.single_label in s_target:
                return torch.tensor(1, dtype=torch.float32)
            else:
                return torch.tensor(0, dtype=torch.float32)

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
        #label = torch.tensor(d[1:].tolist(), dtype=torch.float32)
        # xray = np.array([xray])

        if self.transform:
            image = self.transform(image)

        return image.float(), xray


class XRAY(object):
    def __init__(self, train_image_transform, test_image_transform, force_pre_process=False):
        self.train_set = XrayDataset(csv_file='Data_Entry_2017_v2020.csv', root_dir='/data/matan/nih', train=True,
                                     transform=train_image_transform, force_pre_process=force_pre_process)

        # self.train_loader = torch.utils.data.DataLoader(self.train_set,
        #                                                 batch_size=train_batch_size, shuffle=True,
        #                                                 num_workers=train_workers)

        self.test_set = XrayDataset(csv_file='Data_Entry_2017_v2020.csv', root_dir='/data/matan/nih', train=False,
                        transform=test_image_transform)

        # self.test_loader = torch.utils.data.DataLoader(self.train_set,
        #     batch_size=test_batch_size, shuffle=False, num_workers=test_workers)

    def class_to_index(self, _class=None):
        if _class is None:
            return self.train_set.class_to_idx
        else:
            return self.train_set.class_to_idx[_class]
