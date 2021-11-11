# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.optim as optim
import torchvision.transforms as transforms
from utils.model import select_model
# from utils.options import parse_args_function
# from utils.dataset import Dataset
import os
from PIL import Image
import cv2
import math
import json
import GPy  # import GPy package
from matplotlib import pyplot as plt
import pickle

# args = parse_args_function()

"""# Load Dataset"""

# root = args.input_file
root = '/mnt/disks/hs03/Data_Hand/FHAB/Video_files'
skel_root = '/mnt/disks/hs03/Data_Hand/FHAB/Hand_pose_annotation_v1'
hand_annotation_root = '/mnt/disks/hs03/Data_Hand/FHAB/Hand_anno_test'
pretrained_model = "./checkpoints/fhad/model-3000.pkl"
obj_trans_root = '/mnt/disks/hs03/Data_Hand/FHAB/Object_6D_pose_annotation_v1_1'
video_output_root = '/mnt/disks/hs03/Data_Hand/Output/FHAB/video_final'

# mean = np.array([120.46480086, 107.89070987, 103.00262132])
# std = np.array([5.9113948 , 5.22646725, 5.47829601])
load_set = 'test'
root_file = './datasets/fhad_test/'
images = np.load(os.path.join(root_file, 'images-%s.npy' % load_set))
images = images.tolist()
points2ds_init = np.load(os.path.join(root_file, 'points2d_init-%s.npy' % load_set))
points2ds = np.load(os.path.join(root_file, 'points2d-%s.npy' % load_set))
hands_bb = np.load(os.path.join(root_file, 'hand_bb-%s.npy' % load_set))

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])


def showHandJoints(imgInOrg, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''

    # imgIn = np.copy(imgInOrg)
    imgIn = cv2.cvtColor(np.copy(imgInOrg), cv2.COLOR_RGB2BGR)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = 3

    for joint_num in range(gtIn.shape[0]):

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color,
                       thickness=-1)
        else:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color,
                       thickness=-1)

    for limb_num in range(len(limbs)):

        x1 = gtIn[limbs[limb_num][0], 1]
        y1 = gtIn[limbs[limb_num][0], 0]
        x2 = gtIn[limbs[limb_num][1], 1]
        y2 = gtIn[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 500 and length > 1:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn


def calculate_loss(gtr, est):
    return ((gtr[0] - est[0]) ** 2 + (gtr[1] - est[1]) ** 2) ** 0.5


def get_err(gtr, est):
    out = []
    for join_num in range(gtr.shape[0]):
        out.append(calculate_loss(gtr[join_num], est[join_num]).item())
    return out


def extend_bounding_box(bb, width, height, bias=30):
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


def vconcat_resize_first(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = im_list[0].shape[1]
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_first(im_list_v, interpolation=cv2.INTER_CUBIC)


"""# Model"""
use_cuda = True

model_def = 'HandNet'

model2 = select_model(model_def)

if use_cuda and torch.cuda.is_available():
    model2 = model2.cuda()
    model2 = nn.DataParallel(model2, device_ids=[0])

"""# Load Snapshot"""
model2.load_state_dict(torch.load("./checkpoints/fhad_full/model-250.pkl"))

"""# Test"""
print('Begin testing the network...')

i = 0

errors_1 = []
errors_2 = []
errors_image_1 = []
errors_image_2 = []

frame_width = 1920
frame_height = 1783
black = [0, 0, 0]  # ---Color of the border---

output_data = []
for subject in os.listdir(root):
    for action_name in os.listdir(os.path.join(obj_trans_root, subject)):
        action_path = os.path.join(root, subject, action_name)
        for sq in os.listdir(action_path):
            if sq == '3':   # if it is sequence test
                n_image = 0
                error_action_1 = []
                error_action_2 = []
                sq_name = subject + '_' + action_name + "_" + sq
                video_name = subject + '_' + action_name + "_seq_" + sq + ".avi"
                video_output = cv2.VideoWriter(os.path.join(video_output_root, video_name),
                                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                               20, (frame_width, frame_height))
                for file_name in sorted(os.listdir(os.path.join(action_path, sq, 'color'))):
                    frame_idx = int(file_name.split('.')[0].split('_')[1])

                    # get the inputs
                    file_path = os.path.join(root, subject, action_name, sq, 'color', file_name)

                    if file_path in images:
                        ind = images.index(file_path)

                        # print(f"{i} : {file_path}")
                        # i += 1
                        n_image += 1
                        image = Image.open(file_path)
                        width, height = image.size

                        inputs = transform(image)
                        inputs = torch.unsqueeze(inputs, 0)

                        hand_bb = hands_bb[ind]
                        hand_bb = extend_bounding_box(hand_bb, width, height)

                        x, y, w, h = hand_bb
                        image_hand = image.crop((x, y, x + w, y + h))
                        inputs2 = transform(image_hand)
                        inputs2 = torch.unsqueeze(inputs2, 0)

                        xy_bb = np.full((21, 2), [x, y])

                        labels2d = points2ds[ind]
                        labels2d = torch.tensor(labels2d)
                        labels2d = Variable(labels2d)

                        # est1 = outputs2d[0][:21].cpu().detach().numpy()
                        est1 = points2ds_init[ind]
                        outputs2d = torch.tensor(est1)
                        outputs2d = Variable(outputs2d)
                        if use_cuda and torch.cuda.is_available():
                            outputs2d = outputs2d.float().cuda(device=0)
                            labels2d = labels2d.float().cuda(device=0)


                        point2d = est1 - xy_bb
                        point2d = torch.tensor(point2d)
                        point2d = torch.unsqueeze(point2d, 0)
                        point2d = Variable(point2d)
                        if use_cuda and torch.cuda.is_available():
                            point2d = point2d.float().cuda(device=0)
                            inputs2 = inputs2.float().cuda(device=0)

                        output2d_final_crop = model2(inputs2, point2d)
                        output2d_final = output2d_final_crop.cpu().detach().numpy()
                        output2d_final = output2d_final + xy_bb

                        output2d_final = torch.tensor(output2d_final)
                        output2d_final = Variable(output2d_final)
                        if use_cuda and torch.cuda.is_available():
                            output2d_final = output2d_final.float().cuda(device=0)

                        error1 = get_err(labels2d, outputs2d)
                        error2 = get_err(labels2d, output2d_final[0])

                        error_action_1.extend(error1)
                        error_action_2.extend(error2)

                        errors_image_1.append(np.mean(error1))
                        errors_image_2.append(np.mean(error2))

                        errors_1.extend(error1)
                        errors_2.extend(error2)

                        obj = {
                            'file': file_path,
                            'hand2d_gt': labels2d.cpu().detach().numpy(),
                            'hand2d_pred_HOPE-Net': est1,
                            'hand2d_pred_proposed': output2d_final[0].cpu().detach().numpy()
                        }
                        output_data.append(obj)
                        # show hand joints on the frame to video
                        labels2d_crop = labels2d.cpu().detach().numpy() - xy_bb
                        labels2d_crop = torch.tensor(labels2d_crop)
                        labels2d_crop = Variable(labels2d_crop)
                        if use_cuda and torch.cuda.is_available():
                            labels2d_crop = labels2d_crop.float().cuda(device=0)
                        groudtruth_image = showHandJoints(image_hand, labels2d_crop)
                        hopenet_image = showHandJoints(image_hand, point2d[0])
                        proposed_image = showHandJoints(image_hand, output2d_final_crop[0])
                        constant = cv2.copyMakeBorder(cv2.resize(groudtruth_image, (633, 633)), 10, 60, 0, 10,
                                                      cv2.BORDER_CONSTANT, value=black)
                        constant1 = cv2.copyMakeBorder(cv2.resize(hopenet_image, (633, 633)), 10, 60, 0, 0,
                                                       cv2.BORDER_CONSTANT, value=black)
                        constant2 = cv2.copyMakeBorder(cv2.resize(proposed_image, (633, 633)), 10, 60, 10, 0,
                                                       cv2.BORDER_CONSTANT, value=black)

                        height, width, ch = constant.shape
                        cv2.putText(constant, 'Ground truth', (int(0.25 * width), height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, 0)
                        cv2.putText(constant1, 'HOPE-Net', (int(0.25 * width), height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 3, 0)
                        cv2.putText(constant2, 'Proposed', (int(0.25 * width), height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 3, 0)

                        im1 = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2BGR)
                        im_tile_resize = concat_tile_resize([[im1],
                                                             [constant, constant1, constant2]])
                        video_output.write(im_tile_resize)

                video_output.release()
                print(f"{i} : {sq_name}    error 1: {np.mean(error_action_1)}     max 1: {max(error_action_1)}    min 1: {min(error_action_1)}  n_image = {n_image}")
                print(f"{i} : {sq_name}    error 2: {np.mean(error_action_2)}     max 2: {max(error_action_2)}    min 2: {min(error_action_2)}")
                i += 1

print(f"Total :      error 1: {np.mean(errors_1)}     error 2: {np.mean(errors_2)}    max 1: {max(errors_1)}    min 1: {min(errors_1)}    max 2: {max(errors_2)}    min 2: {min(errors_2)}")
print(f"             error images 1: {np.mean(errors_image_1)}     error images 2: {np.mean(errors_image_2)}    max 1: {max(errors_image_1)}    min 1: {min(errors_image_1)}    max 2: {max(errors_image_2)}    min 2: {min(errors_image_2)}")
with open('datasets/fhad_full/test_output.pkl', 'wb') as f:
    pickle.dump(output_data, f)
