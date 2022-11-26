import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

img_ext = ['.jpg']
vid_ext = ['.mp4']

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--input_path', default='',
                        help='path where input is')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def prediction_text(img, point, predict_cnt):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # point
    point = (int(point[0]), int(point[1]))

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    text = 'Estimated Crowd: ' + str(predict_cnt)
    return cv2.putText(img, text, point , font, fontScale, color, thickness, cv2.LINE_AA)

def circling(model, transform, device, img_path, img):
    if img_path:
        img_raw = Image.open(img_path).convert('RGB')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img)

    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    point = [new_width - new_width*0.1, new_height - new_height*0.2]
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    
    img_to_draw = prediction_text(img_to_draw, point, predict_cnt)

    return img_to_draw

def video_circling(model, transform, device, vid_path, out_dir):
    cap = cv2.VideoCapture(vid_path)
    vid_name = os.path.join(out_dir, vid_path.split('/')[-1])
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3)) // 128 * 128
    frame_height = int(cap.get(4)) // 128 * 128
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(vid_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame_to_draw = circling(model, transform, device, False, frame)
            # Write the frame into the file 'output.avi'
            out.write(frame_to_draw)
        else:
            break
    cap.release()
    out.release()

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    input_path = args.input_path
    # load the images
    # save the visualized image
    if os.path.isdir(input_path):
        for idx, img in enumerate(os.listdir(input_path)):
            img = os.path.join(input_path, img)
            img_to_draw = circling(model, transform, device, img)
            cv2.imwrite(os.path.join(args.output_dir, 'output_{}.jpg'.format(idx)), img_to_draw)
    else:
        if os.path.splitext(input_path)[1] in img_ext:
            img_to_draw = circling(model, transform, device, input_path)
            cv2.imwrite(os.path.join(args.output_dir, 'output.jpg'), img_to_draw)
        elif os.path.splitext(input_path)[1] in vid_ext:
            video_circling(model, transform, device, input_path, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)