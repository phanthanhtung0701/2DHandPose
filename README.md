# Hand segmentation and Hand Pose Estimation

## Hand segmentation with detectron2
1. Download and setup detectron2
2. Copy file on datasets into datasets folder of detectron2 folder
3. Change default train config _C.INPUT.RANDOM_FLIP from "horizontal" to "none" at detectron2/detectron2/config/defaults.py
4. With data DHY, create json file from data to prepare for training, see in file create_json_from_mask_image
5. Training in file hand_segment_train

## Hand Pose Estimation
