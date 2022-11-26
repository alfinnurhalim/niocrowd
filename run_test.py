import argparse

from tqdm import tqdm,trange
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

def draw_text(img, predict_cnt):

    h,w,_ = img.shape

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # point
    point = (int(w*0.01),int(h*0.08))

    # fontScale
    fontScale = 1.5 * (w/h)/(1280/720)

    # Blue color in BGR
    color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    text = 'Crowd: ' + str(predict_cnt)
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
    h,w,_ = img.shape

    # invert Y axis
    list_y = [h-y for y in list_y]

    x = np.array(list_x)
    y = np.array(list_y)

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[0:w:x.size**0.7*1j,0:w:y.size**0.7*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    return xi,yi,zi

def get_heatmap(x,y,img,dpi=100,alpha=0.6):
    h,w,_ = img.shape

    xi,yi,zi = get_density_map(x,y,img)

    # plt.figure(figsize=(w/dpi, h/dpi),dpi=dpi)
    plt.contourf(xi, yi, zi, alpha=alpha)
    plt.imshow(img,extent=[0,w,0,h])
    plt.axis('off')

    figure = plt.gcf()
    figure.canvas.draw()

    b = figure.axes[0].get_window_extent()

    heatmap = np.array(figure.canvas.buffer_rgba())
    # heatmap = heatmap[min(y):max(y),min(x):max(x),:]
    heatmap = heatmap[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]

    heatmap_h,heatmap_w,_ = heatmap.shape
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2BGRA)
    heatmap = cv2.resize(heatmap,(w,h))

    plt.close()

    return heatmap[:,:,:-1]

def process(img,model,threshold,device='cuda',stack=False):
    output = dict()

    # convert RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # preproces image 
    img_input = preprocess_img(device, img_rgb)

    # Run Inference
    outputs = model(img_input)

    # Decode
    x,y = decode_output(outputs,threshold,img)
    heatmap = get_heatmap(x,y,img)

    # draw
    img_dot = img.copy()
    for p in zip(x,y):
        img_dot = cv2.circle(img_dot, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
    img_dot = draw_text(img_dot,len(x))

    # set output
    output['dot'] = img_dot
    output['heatmap'] = heatmap

    if stack:
        canvas = np.hstack((img,img_dot, heatmap))
        output['stack'] = canvas

    return output


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
    output_dir = args.output_dir

    stack = True

    # read img
    if os.path.splitext(input_path)[1] in img_ext:
    
        img = cv2.imread(input_path)
        output = process(img,model,args.thr,device,stack=stack)
        cv2.imwrite(os.path.join(output_dir, 'output.jpg'), output['stack'])

    # read video
    elif os.path.splitext(input_path)[1] in vid_ext:
        cap = cv2.VideoCapture(input_path)
        vid_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir,vid_name+'_output.mp4')

        fps = int(cap.get(5))
        length = int(cap.get(7))

        for i in trange(50):
            ret, frame = cap.read()

            if ret:
                output = process(frame,model,args.thr,device,stack=stack)

                if i==0 :
                    task = 'stack' if stack else 'heatmap'
                    frame_height,frame_width,_ = output[task].shape
                    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

                out.write(output[task])

        cap.release()
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)