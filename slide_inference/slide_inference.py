#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:16:53 2022

@author: user01
"""
import os, argparse, warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
'''
Usage:
    
required arguments:
    -h, --help            
        show this help message and exit. 
        Only model path and video path are required. 
    -m  --model_path
        path to the directory where all the images of sequence are.
    -sd --slide_dir
        directory where slides are.
    -dd --dest_dir
        directory where to write predictions.
    -b --blend
         Whether to overlay predictions over image or not.
    -r --draw_bdr
        Whether to draw borders or fill the nuclei detections..
  
'''
def str2bool(v):
    '''
    Python < 3.9:
        For these versions this function will allow argparset to take Boolean 
        function as an argument.
    Python >= 3.9:
        Following argument can be passed to the argparser to take in boolen argument.
        e.g.
        parser.add_argument('--feature', action=argparse.BooleanOptionalAction)
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# positional required args
parser.add_argument("-m", "--model_path", required=True,
                    help="full path to tf/keras trained model.", type=str)
parser.add_argument("-sd", "--slide_dir", required=True,
                    help="directory where slides are.", type=str)
parser.add_argument("-dd", "--dest_dir", required=True,
                    help="directory where to write predictions.", type=str)
parser.add_argument("-b", "--blend", default=False, type=str2bool, nargs='?',
                    const=True, help="Whether to overlay predictions over image or not.")
parser.add_argument("-r", "--draw_bdr", default=True, type=str2bool, nargs='?',
                    const=True, help="Whether to draw borders or fill the nuclei detections.")

args = parser.parse_args()

model_path = args.model_path
model_path = model_path.replace('\\', '/') 

slide_dir = args.slide_dir
slide_dir = slide_dir.replace('\\', '/') 

dest_dir = args.dest_dir
dest_dir = dest_dir.replace('\\', '/') 

BLEND = args.blend
DRAW_BDR = args.draw_bdr

import tensorflow as tf
tf.autograph.set_verbosity(level=3, alsologtostdout=False)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("[INFO] Using GPU.")
elif tf.test.is_built_with_cuda() == False:
    warnings.warn("Current Tensorflow/Keras installed don't support CUDA operation.\
                  Using CPU, processing will be slow.")
else:
    warnings.warn("GPU not detected using CPU, processing will be slow.")
    
from utils import (Tumor_IO, read_img, bwmorph_thin, gray2encoded, seprate_instances, water, 
                   remove_small_obj_n_holes, assgin_via_majority, decode_predictions, my_argmax,
                   get_inst_seg, get_sem, get_inst_seg_bdr, get_sem_bdr)
from gray2color import gray2color
from fmutils import fmutils as fmu
import numpy as np
import cv2
from tqdm import trange


img_paths = fmu.get_all_files(slide_dir)

print('[INFO] Loading model...')
model = tf.keras.models.load_model(filepath=model_path, compile=False) 
print('=> Model loaded successfully')
# print(f'Input: {model.input}')
# print(f'Ontputs: {model.output}')
#%
i = 0

for i in trange(len(img_paths), desc='Generating Predictions'):
    img_rgb, h = read_img(img_paths[i], 512, 512)
    name = fmu.get_basename(img_paths[i], False)
    img = np.concatenate((img_rgb, h), axis=-1)
    img = img[np.newaxis, :,:,:]
    
    
    _, seg_op, inst_op = model.predict(img)
    
    seg_op, inst_op = decode_predictions(seg_op, inst_op)
    
    pred_sep_inst = seprate_instances(seg_op, inst_op, 6, True, 3).astype(np.uint8)
    
    if DRAW_BDR:
        inst = get_inst_seg_bdr(pred_sep_inst, img_rgb, blend=BLEND)
        sem = get_sem_bdr(seg_op, img_rgb, blend=BLEND)
    else:
        inst = get_inst_seg(pred_sep_inst, img_rgb, blend=BLEND)
        sem = get_sem(seg_op, img_rgb, blend=BLEND)

    cv2.imwrite(f'{dest_dir}inst_{name}.png', cv2.cvtColor(inst, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'{dest_dir}sem_{name}.png', cv2.cvtColor(sem, cv2.COLOR_BGR2RGB))
