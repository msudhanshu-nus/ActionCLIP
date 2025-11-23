# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class MultiLabelVideoRecord(object):
    """
    For lines like:
    <path> <num_frames> <y_bleed> <y_mech> <y_therm>
    e.g.
    MultiBypass_frames/BBP01_clip01  80  1 0 0
    """

    def __init__(self, row):
        # row is a list of strings from x.strip().split()
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        # Use the first 3 flags: bleeding, mechanical, thermal. Strip bracket wrappers like "[0]".
        tokens = self._data[2:5]
        if len(tokens) != 3:
            raise ValueError(f"Expected 3 label tokens, got {len(tokens)} from row: {self._data}")

        labels = []
        for token in tokens:
            cleaned = token.strip().strip("[]")
            try:
                labels.append(float(cleaned))
            except ValueError:
                raise ValueError(f"Could not parse label token '{token}' from row: {self._data}")
        return np.array(labels, dtype=np.float32)

    @property
    def severity(self):
        # Severity levels for bleeding, mechanical, thermal (tokens 5,6,7 in the txt row).
        tokens = self._data[5:8]
        if len(tokens) != 3:
            raise ValueError(f"Expected 3 severity tokens, got {len(tokens)} from row: {self._data}")

        levels = []
        for token in tokens:
            cleaned = token.strip().strip("[]")
            try:
                levels.append(int(cleaned))
            except ValueError:
                raise ValueError(f"Could not parse severity token '{token}' from row: {self._data}")
        return np.array(levels, dtype=np.int64)

class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:08d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1, include_severity=False):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.include_severity = include_severity
        self.frame_cache = {}  # directory -> sorted list of frame filenames

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _get_frame_list(self, directory):
        """Return a cached, lexicographically sorted list of frame filenames in a directory."""
        if directory not in self.frame_cache:
            files = [
                f for f in os.listdir(directory)
                if f.lower().endswith(".jpg") or f.lower().endswith(".png")
            ]
            files.sort()
            if len(files) == 0:
                raise FileNotFoundError(f"No image frames found in directory: {directory}")
            self.frame_cache[directory] = files
        return self.frame_cache[directory]

    def _load_image(self, directory, idx):
        """
        Load image by indexing into the sorted frame list for the directory.
        idx is biased by self.index_bias (default 1), so convert to 0-based before lookup.
        """
        frame_list = self._get_frame_list(directory)
        zero_based = idx - self.index_bias
        zero_based = max(0, min(zero_based, len(frame_list) - 1))
        fname = frame_list[zero_based]
        return [Image.open(os.path.join(directory, fname)).convert('RGB')]
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        with open(self.list_file, "r") as handle:
            self.video_list = [
                MultiLabelVideoRecord(line.strip().split())
                for line in handle
                if line.strip()
            ]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames,
                        size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int64) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int64) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int64) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        target = torch.from_numpy(record.label)
        if self.include_severity:
            severity = torch.from_numpy(record.severity)
            return process_data, target, severity
        return process_data, target

    def __len__(self):
        return len(self.video_list)
