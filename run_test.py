import argparse

import torch
import torchvision.transforms as standard_transforms
import numpy as np

import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde

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
    parser.add_argument('--thr', default=0.5, type=float,
                        help="input model threshold")

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

def preprocess_img(device, img):
    height,width,_ = img.shape

    # resize to match net inputs
    new_width = int(width // 128 * 128)
    new_height = int(height // 128 * 128)

    img = cv2.resize(img,(new_width, new_height))

    # transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # pre-proccessing
    img = transform(img)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)

    return samples

def decode_output(outputs,threshold,img):
    height,width,_ = img.shape

    # resize to match net inputs
    width_ratio = width / int(width // 128 * 128)
    height_ratio = height / int(height // 128 * 128)

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]

    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    points = [[int(x*width_ratio),int(y*height_ratio)] for x,y in points]

    list_x,list_y = zip(*points)

    return list_x,list_y

def get_density_map(list_x,list_y,img):
    h,_,_ = img.shape

    # invert Y axis
    list_y = [h-y for y in list_y]

    x = np.array(list_x)
    y = np.array(list_y)

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.7*1j,y.min():y.max():y.size**0.7*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    return xi,yi,zi

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
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get the P2PNet
    model = build_model(args)
    model.to(device)

    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.eval()

    # set your image path here
    input_path = args.input_path

    # read img
    img_raw = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # preproces image 
    img_input = preprocess_img(device, img_rgb)

    # Run Inference
    outputs = model(img_input)

    # Decode
    x,y = decode_output(outputs,args.thr,img_raw)

    # draw
    canvas = img_raw.copy()

    for p in zip(x,y):
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

    cv2.imshow('sds',canvas)
    cv2.waitKey(0)

    xi,yi,zi = get_density_map(x,y,img_raw)

    plt.contourf(xi, yi, zi, alpha=0.5)

    plt.imshow(img_rgb, extent=[min(x),max(x),min(y),max(y)])
    plt.show()

    # if folder
    # if os.path.isdir(input_path):
    #     for idx, img in enumerate(os.listdir(input_path)):
    #         img = os.path.join(input_path, img)
    #         img_to_draw = circling(model, transform, device, img)
    #         cv2.imwrite(os.path.join(args.output_dir, 'output_{}.jpg'.format(idx)), img_to_draw)

    # # if single img or video
    # else:
    #     if os.path.splitext(input_path)[1] in img_ext:
    #         img_to_draw = circling(model, transform, device, input_path)
    #         cv2.imwrite(os.path.join(args.output_dir, 'output.jpg'), img_to_draw)
    #     elif os.path.splitext(input_path)[1] in vid_ext:
    #         video_circling(model, transform, device, input_path, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)