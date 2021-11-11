from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

import numpy as np
import time
import json
import cv2
import pickle
import os
from PIL import Image
import math
import argparse
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from HAND_model.utils.model import select_model


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


def showHandBoxJoints(imgInOrg, gtIn, bb=None, filename=None, upscale=1, lineThickness=3):
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
    if bb is not None:

        cv2.rectangle(imgIn, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])), (0, 255, 0), 3)
    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn


def calculate_loss(gtr, est):
    return ((gtr[0] - est[0]) ** 2 + (gtr[1] - est[1]) ** 2) ** 0.5


def get_loss(gtr, est):
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


parser = argparse.ArgumentParser()
# Required arguments: input and output files.
parser.add_argument(
    "--input_folder",
    default='/mnt/disks/hs03/Data_Hand/FHAB/Video_files/Subject_4/pour_milk/3/color',
    help="Input image, directory"
)
parser.add_argument(
    "--output_folder",
    default='demo_output',
    help="Prefix of output pkl filename"
)
args = parser.parse_args()

# detect and segment hand predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = "/opt/pycode/detectron2/output_DATN/model_0019999.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
print('Mask R-CNN is loaded')

# Hand pose estimator
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
use_cuda = True
model = select_model('HopeNet')
model2 = select_model('HandNet')

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    model2 = model2.cuda()
    model2 = nn.DataParallel(model2, device_ids=[0])

model.load_state_dict(torch.load('/opt/pycode/Hope/HOPE/checkpoints/fhad_bak/model-3000.pkl'))
model2.load_state_dict(torch.load("/opt/pycode/Hope/HOPE/checkpoints/fhad_full/model-250.pkl"))
path = args.input_folder
outpath = args.output_folder

i = 0
for file_name in sorted(os.listdir(path)):
    file_path = os.path.join(path, file_name)

    # detect hand
    img = cv2.imread(file_path)
    start_time = time.time()

    predictions = predictor(img)
    predictions = predictions["instances"].to("cpu")
    boxes = predictions.pred_boxes.tensor.numpy()
    scores = predictions.scores.numpy()

    if len(boxes) > 0:
        print(f"{i} : {file_name} detect {len(boxes)} hand     {time.time() - start_time}")
        box = boxes[0]
        x0, y0, x1, y1 = box
        hand_bb = [x0, y0, x1 - x0, y1 - y0]
    else:
        print(f"{i} : {file_name} can't detect hand     {time.time() - start_time}")
        continue

    # detect hand pose
    image = Image.open(file_path)
    width, height = image.size

    inputs = transform(image)
    inputs = torch.unsqueeze(inputs, 0)

    # hand_bb_path = os.path.join(hand_bb_root, subject, action, sq)
    # hand_bb = get_hand_bb(hand_bb_path, file_name)
    hand_bb = extend_bounding_box(hand_bb, width, height)

    x, y, w, h = hand_bb
    inputs2 = image.crop((x, y, x+w, y+h))
    inputs2 = transform(inputs2)
    inputs2 = torch.unsqueeze(inputs2, 0)

    xy_bb = np.full((21, 2), [x, y])

    # joints2Dimage = os.path.join(outpath, frame_name)
    joints2Dimage = os.path.join(outpath, file_name)

    # joints2DimageOr = os.path.join(outpath0, file_name)
    # wrap them in Variable
    # Y.append(np.concatenate(labels2d, axis=None))

    # point2d = labels2d - xy_bb
    inputs = Variable(inputs)

    if use_cuda and torch.cuda.is_available():
        inputs = inputs.float().cuda(device=0)

    outputs2d_init, outputs2d, outputs3d = model(inputs)
    # Y_new.append(np.concatenate(outputs2d[0][:21].cpu().detach().numpy(), axis=None))

    est1 = outputs2d[0][:21].cpu().detach().numpy()

    point2d = est1 - xy_bb
    point2d = torch.tensor(point2d)
    point2d = torch.unsqueeze(point2d, 0)
    point2d = Variable(point2d)
    if use_cuda and torch.cuda.is_available():
        point2d = point2d.float().cuda(device=0)
        inputs2 = inputs2.float().cuda(device=0)

    output2d_final = model2(inputs2, point2d)
    end_time = time.time()
    output2d_final = output2d_final.cpu().detach().numpy()
    output2d_final = output2d_final + xy_bb

    output2d_final = torch.tensor(output2d_final)
    output2d_final = Variable(output2d_final)
    if use_cuda and torch.cuda.is_available():
        output2d_final = output2d_final.float().cuda(device=0)

    # showHandJoints(image, output2d_final[0], filename=joints2Dimage)
    showHandBoxJoints(image, output2d_final[0], bb=hand_bb, filename=joints2Dimage)
    # showHandBoxJoints(image, outputs2d[0][:21], bb=hand_bb, filename=joints2Dimage)
    # showHandJoints(image, outputs2d[0][:21], filename=joints2Dimage)

    print(f"{i} : {file_path}   {end_time-start_time}")
    i += 1

