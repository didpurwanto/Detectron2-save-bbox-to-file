# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import hickle as hkl
import math 

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import GenerateLabel

# constants
WINDOW_NAME = "COCO detections"

visualization = False

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",metavar="FILE",)
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+",)
    parser.add_argument("--output",)
    parser.add_argument("--confidence-threshold",type=float,default=0.5,)
    parser.add_argument("--opts",default=[],nargs=argparse.REMAINDER,)
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    cfg = setup_cfg(args)
    demo = GenerateLabel(cfg)
    fconf = open(args.output+'.txt',"w")

    if(visualization):
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    boxes = np.array([])
    
    for path in tqdm.tqdm(args.input, disable=not args.output):
        out_filename = os.path.join(args.output, os.path.basename(path))
        
        img = read_image(path, format="BGR")
        img_shape = img.shape

        # save visualization
        if(visualization):
            bbox, visualized_output = demo.save_bbox(img, img_shape)
            visualized_output.save(out_filename)
        else:
            bbox = demo.save_bbox_novisual(img, img_shape)

        # save bbox in .txt
        fconf.write("%s %s %s %s\n" % (str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))

        # save bbox in .h5 file
        if boxes.shape[0] == 0:
            boxes = bbox
        else:
            boxes = np.vstack((boxes, bbox))

    fconf.close()
    hkl.dump(boxes, args.output + '.h5')