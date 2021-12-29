import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
import random
import tensorflow as tf
from pallet_n_classnames import pallet_ADE20K, pallet_cityscape, pallet_VOC, pallet_mine, pallet_vistas
from models import  num_of_classes
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 700

num_class = num_of_classes()

def shape_corrector(img):
    start = img.squeeze()
    start = start.transpose(2,0,1)
    j = 5
    num_c,w,h = start.shape
    new = np.empty((0,w,h))
    for i in range(num_c):
        tempx = start[i,:,:]
        tempy = tempx*j
        j = j + 4
        new = np.append(new, [tempy], axis=0)
    final = np.sum(new, axis=0)
    return final

def gray_2_rgb(gray_processed, pallet):
    
    gray = gray_processed
    w, h = gray.shape
    gray = gray[:,:,np.newaxis]
    gray = tf.image.grayscale_to_rgb((tf.convert_to_tensor(gray)))
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        gray = sess.run(gray)
    else:
            gray = gray.numpy()
    gray = tf.cast(gray, 'int32')
    if tf.executing_eagerly()==False:
        sess1 = tf.compat.v1.Session()
        gray = sess1.run(gray)
    else:
            gray = gray.numpy()
    unq = np.unique(gray)
    rgb = np.zeros((w,h,3))
    
    for i in range(len(unq)):
        #clr = pallet[:,i:i+1,:]
        clr = pallet[:,unq[i],:]
        rgb = np.where(gray!=unq[i], rgb, np.add(rgb,clr))
        
        if tf.executing_eagerly()==False:
            sess1 = tf.compat.v1.Session()
            rgb = sess1.run(rgb)
        else:
            rgb = rgb.numpy()
            
    return rgb

def tf_gray2rgb(gray_processed, pallet):
    
    pallet = tf.convert_to_tensor(pallet)
    w, h = gray_processed.shape
    gray = gray_processed[:,:,np.newaxis]
    gray = tf.image.grayscale_to_rgb((tf.convert_to_tensor(gray)))
    gray = tf.cast(gray, 'int32')
    unq = np.unique(gray_processed)
    rgb = tf.zeros_like(gray, dtype=tf.float64)
    
    for i in range(len(unq)):
        clr = pallet[:,unq[i],:]
        clr = tf.expand_dims(clr, 0)
        rgb = tf.where(tf.not_equal(gray,unq[i]), rgb, tf.add(rgb,clr))
        
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        rgb = sess.run(rgb)
    else:
        rgb = rgb.numpy()
    return rgb

def gt_corrector(gt):
    gt = gt #/ 255
    gt = gt #* (num_class+1)
    gt = tf.cast(gt, "int32")
    gt = tf.one_hot(gt, num_class, axis=-1)
    
    if tf.executing_eagerly()==False:
        sess1 = tf.compat.v1.Session()
        gt = sess1.run(gt)
    else:
        gt = gt.numpy()
        
    gt = np.argmax(gt, 2)
    return gt


def sigmoid_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.sigmoid(pred)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def softmax_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.softmax(pred, axis=-1)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def plot_results(path, im_height, im_width, model, show_img = 5, clr_map='gray', Threshold = 0.5, 
                 pallet=pallet_ADE20K, R_seed = 5, gray2rgb=False, clr_masks = True, activation=True):
    
    show_img = show_img
    img_extension = '.jpg'#  png,jpg
    mask_extension = '.png'
    path_images = path + '/images/images/'
    random.seed(R_seed)
    names = []
    for i in range(show_img):
        n = random.choice([x for x in os.listdir(path_images) 
                           if os.path.isfile(os.path.join(path_images, x))]).rstrip(img_extension)
        names.append(np.asarray(n))    
    names = np.asarray(names)
        
    if  gray2rgb:
        mask_flag = 1    #0 if masks are in grayscale
    else:
        mask_flag = 0
        
    if clr_masks == False:
        mask_flag = 0
        
    imgs = []
    masks = []
    gts = []
    
    for i in range(show_img):
        
        img = cv2.imread(path+'/images/images/' + names[i] + '.jpg')
        if img is None:
            img = cv2.imread(path+'/images/images/' + names[i] + '.png')
        img = cv2.resize(img, (im_height, im_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(path+'/masks/masks/' + names[i] + '.png', mask_flag)
        if gt is None:
            gt = cv2.imread(path+'/masks/masks/' + names[i] + '.jpg', mask_flag)
        gt = cv2.resize(gt, (im_height, im_width), interpolation=cv2.INTER_NEAREST)
        
        if gray2rgb==True and clr_masks==True:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gts.append(np.asarray(gt))
        elif gray2rgb==True and clr_masks==False:
            print('Coloring grayscale GT/labels, wait ...')
            gt = gt_corrector(gt)
            gt = tf_gray2rgb(gt, pallet)
            gts.append(np.asarray(gt))
        else:
            gts.append(np.asarray(gt))
            
        imgs.append(np.asarray(img))

    imgs = np.asarray(imgs)
    gts = np.asarray(gts)
    
    pred = model.predict(imgs/255, verbose=1)#Normalizing
    if activation == True:
        if pred.shape[-1] < 2:
          pred = sigmoid_activation(pred)
        else:
          pred = softmax_activation(pred)
    pred = (pred > Threshold).astype(np.uint8)
    _,_,_,b_shape = pred.shape
    if b_shape<=2:
        pred = pred[:,:,:,0]
        clr_map2 = clr_map +'_r' # for boundaries
    else:
        clr_map2 = clr_map
        if gray2rgb:
            print('Coloring Predictions, wait...')
            for i in range(len(names)):
                temp = np.argmax(pred[i,:,:,:], 2)
                temp = tf_gray2rgb(temp, pallet)
                masks.append(np.asarray(temp))
                print(i+1, 'Image(s) Done')
            pred = np.asarray(masks)
        else:
            for i in range(len(names)):
                temp = np.argmax(pred[i,:,:,:], 2)
                masks.append(np.asarray(temp))
            pred = np.asarray(masks)
    plot_row = 3
    plot_col = show_img
    fig, axs = plt.subplots(plot_row, plot_col, figsize = (int(show_img*3+3), 10))
    #fig.suptitle('From Topto Bottom Original_Img, Ground_Truth, Predicted')
    for row in range(plot_row):
        for col in range(plot_col):
            if row==0:   
                axs[row,col].imshow(imgs[col], cmap = clr_map, interpolation = 'bilinear')
                axs[row,col].axis("off")
                axs[row,col].set_title('Image')
            if row==1:
                axs[row,col].imshow(gts[col], cmap = clr_map, interpolation = 'bilinear')
                axs[row,col].axis("off")
                axs[row,col].set_title('Ground_Truth')
            if row==2:
                axs[row,col].imshow(pred[col], cmap = clr_map2, interpolation = 'bilinear')#+'_r'
                axs[row,col].axis("off")
                axs[row,col].set_title('Predictions')
   
    return fig

