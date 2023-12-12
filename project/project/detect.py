#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

"""
Demo of running YOLOv5 for sign detection on an image.  
Code simplified from detect.py in YOLOv5

Requires cloning of YOLOv5 repo: https://github.com/ultralytics/yolov5
"""

import os
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import torch
from pathlib import Path
import platform
import argparse
import sys
import glob

from std_msgs.msg import Bool

# YOLOv5 setup.  Adjust this to point to your yolov5 folder:
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3] / 'yolov5'  # YOLOv5 root directory -- adjust this
if not ROOT.is_dir():
    raise ValueError(f"No yolov5 repo located here: {ROOT}")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from ultralytics.utils.plotting import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

CONF_THRESHOLD = 0.4

class Detect(Node):
    def __init__(self, weights):
        super().__init__('crosswalks')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.run_detect, 10)

        self.publisher = self.create_publisher(
            Bool, '/crosswalk', 10)

        print('Loading model')
        imgsz = np.array([640,640])
        device = select_device('')
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='coco128.yaml', fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

    def run_detect(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        img_resized = cv2.resize(img,self.imgsz)  # convert to target size
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # convert to RGB
        img_tensor = torch.from_numpy(img_resized).to(self.model.device)
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # Pytorch needs format: channels x height x width

       
        pred = self.model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=0.45)

        det = pred[0]
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()

        msg = Bool()
        msg.data = False

        annotator = Annotator(img, line_width=1, example=str(self.names))
        for *xyxy, conf, cls in reversed(det):
            msg.data = True
            label = f'{self.names[int(cls)]} {conf:.1f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))

        self.publisher.publish(msg)
        self.get_logger().info('Crosswalk: "%s"' % msg.data)

        imout = annotator.result()
        cv2.imshow('Detection', imout)
        cv2.waitKey(1)

def run_on_image(weights, # .pt file
             imgname, # input image name
             imgsz = (256,256), # inference size (height, width)
             device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             ):

    # Load model
    print('Loading model')
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data='coco128.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Read image
    print(f'Reading image {imgname}')
    im0 = cv2.imread( imgname )
    if im0 is None:
        print('Warning, unable to read:', imgname)
    img = cv2.resize(im0,imgsz)  # convert to target size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = img.transpose(2,0,1)  # Pytorch needs format: channels x height x width
    img = np.ascontiguousarray(img)

    # convert image to Pytorch tensor
    im = torch.from_numpy(img).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    im = im[None]  # Add extra dimension for convolutions

    print(f'Input image shape: {im.shape}' )
    
    # Inference
    print('Doing inference')
    pred = model(im)

    # NMS
    print('Doing NMS')
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    print('Processing predictions')
    # Process predictions
    det = pred[0]   # Single image so single prediction
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

    print('Creating annotator')    
    annotator = Annotator(im0, line_width=1, example=str(names))
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

    print('Showing output')
    imout = annotator.result()
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow('Detection', imout.shape[1], imout.shape[0])
    cv2.imshow('Detection', imout)
    cv2.waitKey()  # Press any key when complete    
    
    cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    path = str(FILE.parents[2] / 'runs' / 'train' / 'exp3' / 'weights' / 'best.pt')
    node = Detect(weights=path)
    #run_on_image(weights=path, imgname=str(FILE.parents[2] / 'test_img' / 'test_img5.jpg'))
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

