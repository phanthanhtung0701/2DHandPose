# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import json
import torch.utils.data as data
from PIL import Image
import torch

"""# Load Dataset"""


class Custom_Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None, method=1):
        self.root = root  # os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        self.method = method

        self.images = np.load(os.path.join(root, 'images-%s.npy' % self.load_set))
        self.points2d_init = np.load(os.path.join(root, 'points2d_init-%s.npy' % self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy' % self.load_set))
        self.hand_bb = np.load(os.path.join(root, 'hand_bb-%s.npy' % self.load_set))

        # if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image = Image.open(self.images[index])
        width, height = image.size
        _hand_bb = self.hand_bb[index]
        hand_bb = self.extend_bounding_box(_hand_bb, width, height)
        x, y, w, h = hand_bb
        crop_hand = image.crop((x, y, x+w, y+h))

        xy_bb = np.full((21, 2), [x, y])
        points2d_init = self.points2d_init[index]

        if self.method == 2:
            xy_bb_hand = np.full((21, 2), points2d_init[0])
            points2d_init = points2d_init - xy_bb_hand
        else:
            points2d_init = points2d_init - xy_bb

        point2d = self.points2d[index]
        point2d = point2d - xy_bb

        if self.transform is not None:
            crop_hand = self.transform(crop_hand)

        hand_bb = torch.tensor(hand_bb)
        return crop_hand[:3], hand_bb, points2d_init, point2d

    def __len__(self):
        return len(self.images)

    def extend_bounding_box(self, bb, width, height, bias=30):
        x, y, w, h = bb
        if x - bias >= 0:
            x = x - bias
            w = w + bias
        else:
            w = w + x
            x = 0

        if y - bias >= 0:
            y = y - bias
            h = h + bias
        else:
            h = h + y
            y = 0

        if x + w + bias <= width:
            w = w + bias
        else:
            w = width - x

        if y + h + bias <= height:
            h = h + bias
        else:
            h = height - y

        return [x, y, w, h]
