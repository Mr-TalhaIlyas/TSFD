import numpy as np
from scipy import ndimage as ndi
import tensorflow as tf
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from plotting import tf_gray2rgb, sigmoid_activation, softmax_activation
from skimage.segmentation import find_boundaries
from gray2color import gray2color
from pallet_n_classnames import pallet_pannuke
import cv2
import copy
from gray2color import gray2color
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 700
"""
    Perform morphological thinning of a binary image
    
    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.
    
    n_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.
    
    Returns
    -------
    out : ndarray of uint8
        Thinned image.
    
    See also
    --------
    skeletonize
    
    Notes
    -----
    This algorithm [1]_ works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.
"""
# lookup tables for bwmorph_thin
G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool)

def bwmorph_thin(image, n_iter=None):
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:  
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    skel = skel.astype(np.bool)
    return skel.astype(np.uint8)
    
def gray2encoded(y_true, num_class):
    '''
    Parameters
    ----------
    y_true : 2D array of shape [H x W] containing unique pixel values for all N classes i.e., [0, 1, ..., N] 
    num_class : int no. of classes inculding BG
    Returns
    -------
    encoded_op : one-hot encoded 3D array of shape [H W N] where N=num_class

    '''
    num_class = num_class
    
    y_true = tf.cast(y_true, 'int32')
    
    encoded_op = tf.one_hot(y_true, num_class, axis = -1)
    
    if tf.executing_eagerly()==False:
        sess1 = tf.compat.v1.Session()
        encoded_op = sess1.run(encoded_op)
    else: 
        encoded_op = encoded_op.numpy()
    return encoded_op

def seprate_instances(sem_mask, instance_boundaries, num_classes, apply_morph=True, kernel_size=3):
    '''

    Parameters
    ----------
    sem_mask : 2D array of shape [H x W] containing unique pixel values for all N classes i.e., [0, 1, ..., N]
    instance_boundaries : 2D array of shape [H x W] bounderies for all N classes i.e., [0->BG, 1->boundry]
    num_classes : no of classes in the sem mask including BG an int
    apply_morph : apply morphological operator so that the edges which were chipped of will be recovered
    Returns
    kernel_size : int kernel size to apply morphological operations (3 default b/c gives best results)
    -------
    op : 3D array containing seperated instances in each channel shape [H x W x N]

    '''
    
    # change datatypt to perform operation
    instances = instance_boundaries.astype(np.float16)
    sem_mask = sem_mask.astype(np.float16)
    instances2 = instances * 6 # bc largest value in sem mask is 5
    
    t = np.subtract(sem_mask, instances2)
    negative_remover = lambda a: (np.abs(a)+a)/2 # one line funstion created by lamda 1 input and 1 output
    t = negative_remover(t).astype(np.uint8)
    # or you can use following line
    #t = np.where(t > 0, t, 0).astype(np.uint8)
    
    # Now as in PanNuke dataset the BG was in 5ht channel and during preprocessing we shifted it to 
    # 0th channel. Now going back so that 0th channel is Neoplastic class and 5th channel is BG as given 
    # in original data description.
    
    if len(np.unique(cv2.fastNlMeansDenoising(t))) == 1:# 1st denoising there might be some noise in the op image
        # if only BG is present than only last channel will be one, do it here
        # b/c the np where conditions wont have any effect on the array if it 
        # only have one class
        tt = np.zeros((t.shape[0], t.shape[1], num_classes))
        tt[:,:,5] = tt[:,:,-1] + 1
        t = tt
    else:# if have atleast one nuclie present/ swaping channels again to match GT
        t = np.where(t == 5, 6, t)
        t = np.where(t == 0, 5, t)
        t = np.where(t == 6, 0, t)
        
        t = gray2encoded(t, num_classes)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))# before i started main_203 it was 2x2
    op = np.zeros(t.shape)
    for i in range(num_classes):
        # Bc at some place boundry is diagonal and very thin (1px) so measure-label
        # will join two seprate blobs so this will seprate them a little
        t[:,:,i] = cv2.erode(t[:,:,i],kernel,iterations = 1)
        # b/c now 5th channel is BG; still 0 digit represents BG in all channels
        # in 5th channel also the BG of the BG*
        op[:,:,i] = measure.label(t[:,:,i], connectivity=2, background=0)# 2 is ususal
        
    if apply_morph == True:
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for i in range(num_classes-1):# bc last channel has BG we dont want to change that    
            op[:,:,i] = cv2.dilate(op[:,:,i],kernel,iterations = 1)
            
    op[:,:,5] = np.where(op[:,:,5]>1, 1, op[:,:,5])
     
    return op




def remove_small_obj_n_holes(seg_op, min_area=10, kernel_size=3):
    '''
    Parameters
    ----------
    seg_op :  a 4D array of N channels [1 H W N] where N is number of classses
    min_area : The smallest allowable object size.
    kernel_size : int kernel size to apply morphological operations (3 default b/c gives best results)
    Returns
    -------
    a : 4D array of N channels [1 H W N] with noise removed and holes filled
    '''
    seg_op = copy.deepcopy(seg_op).astype(np.uint8)
    #k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    a = seg_op.squeeze()
    for i in range(a.shape[-1]-1): # iterate over each class seprately
        # need to convert array into boolen type
        b = morphology.remove_small_objects(a[:,:,i+1].astype(bool), min_size=min_area).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        a[:,:,i+1] = b
        #a[:,:,i+1] = morphology.convex_hull_object(b, connectivity=2)
        #a[:,:,i+1] = binary_fill_holes(b).astype(int)
    a = a[np.newaxis,:,:,:]# keep IO size consistant
    
    return a

def assgin_via_majority(seg):
    '''
    Parameters
    ----------
    seg : 2D array containing unique pixel values for each class
    Returns
    -------
    x: 2D array where an instance is assigned to be the class of most frequently
       occuring pixel value (as each unique pixel value represent a class).
    '''
    a = copy.deepcopy(seg).astype(np.uint8)
    # 1st convert to binary mask
    _, th = cv2.threshold(a, 0, 1, cv2.THRESH_BINARY)
    # now measure label
    b = measure.label(th, connectivity=2, background=0)
    # now make n unique channels n= no. of labels measured
    c = gray2encoded(b, len(np.unique(b)))
    
    op = np.zeros(c.shape)
    for i in range(len(np.unique(b))-1):
        temp = np.multiply(c[:,:,i+1], a)# multiply each channel element wise
        mfp = most_frequent_pixel(temp)
        # now convert the range form [0, 1] to [0, mfp]
        _, temp = cv2.threshold(temp, 0, mfp, cv2.THRESH_BINARY)
        op[:,:,i+1] = temp
    x = np.sum(op, axis=2)
    
    return x.astype(np.uint8)

def most_frequent_pixel(img):
    '''
    Parameters
    ----------
    img : 2D array containing unique pixel values for each class
    Returns
    -------
    op : int, most frequently occuring pixel value excluding which has pixel value of 0
    '''
    unq, count = np.unique(img, return_counts=True)
    idx = np.where(count == np.max(count[1:]))
    op = int(unq[idx][0])
    
    return op

def decode_predictions(seg_op, inst_op, thresh=0.5):
    '''
    Parameters
    ----------
    seg_op : Raw logits from CNN output, shape [B, H, W, N]
    inst_op : Raw logits from CNN output, shape [B, H, W, 1]
    thresh : Threshold on pixel confidence a float between [0, 1]
    Returns
    -------
    seg_op : activated and thresholded output of CNN
    inst_op : activated and thresholded output of CNN
    '''
    seg_op = softmax_activation(seg_op)
    seg_op = (seg_op > thresh).astype(np.uint8)
    seg_op = remove_small_obj_n_holes(seg_op, min_area=22, kernel_size=3)
    seg_op = np.argmax(seg_op[0,:,:,:], 2).astype(np.uint8)
    seg_op = assgin_via_majority(seg_op) # assigning instance via majority pixels ((post processing))
    seg_op = (seg_op).astype(np.uint8)
    
    inst_op = sigmoid_activation(inst_op)
    inst_op = (inst_op > thresh).astype(np.uint8)
    inst_op = inst_op.squeeze()
    inst_op = (inst_op).astype(np.uint8)
    inst_op = bwmorph_thin(inst_op)
    
    return seg_op, inst_op

def get_inst_seg(sep_inst, img, blend=True):
    '''
    Parameters
    ----------
    sep_inst : a 3D array of shape [H, W, N] where N is number of classes and in
            each channel all the instances have a unique value.
    img : Original RGB image for overlaying the instance seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each instance have of each
            and all classes have a unique RGB value 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    '''    
    # if you get shape mismatch error try swaping the (w,h) argument of the line below.
    # i.e., from (x.shape[0], x.shape[1]) to (x.shape[1], x.shape[0]).
    img = cv2.resize(img, (sep_inst.shape[0], sep_inst.shape[1]), interpolation=cv2.INTER_LINEAR) 
    sep_inst = measure.label(sep_inst[:,:,0:5], connectivity=2, background=0) # ignore BG channel i.e. 6th ch.
    # take element wise sum of all channels so that each instance of each class
    # has a unique value in whole 3D array.
    sep_inst = np.sum(sep_inst, axis=-1) 
    rgb = gray2color(sep_inst, use_pallet='ade20k')
    if blend:
        inv = 1 - cv2.threshold(sep_inst.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, rgb)
    else:
        blend = rgb
    
    return blend

def get_inst_seg_bdr(sep_inst, img, blend=True):
    '''
    Parameters
    ----------
    sep_inst : a 3D array of shape [H, W, N] where N is number of classes and in
            each channel all the instances have a unique value.
    img : Original RGB image for overlaying the instance seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each instance have of each
            and all classes have a unique RGB border. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    # if you get shape mismatch error try swaping the (w,h) argument of the line below.
    # i.e., from (x.shape[0], x.shape[1]) to (x.shape[1], x.shape[0]).
    img = cv2.resize(img, (sep_inst.shape[0], sep_inst.shape[1]), interpolation=cv2.INTER_LINEAR) 
    sep_inst = measure.label(sep_inst[:,:,0:5], connectivity=2, background=0)# ignore BG channel i.e. 6th ch.
    # take element wise sum of all channels so that each instance of each class
    # has a unique value in whole 3D array.
    sep_inst = np.sum(sep_inst, axis=-1)
    # isolate all instances 
    sep_inst_enc = gray2encoded(sep_inst, num_class=len(np.unique(sep_inst)))
    # as the in encoded output the 0th channel will be BG we don't need it so
    sep_inst_enc = sep_inst_enc[:,:,1:]
    # get boundaries of thest isolated instances
    temp = np.zeros(sep_inst_enc.shape)
    for i in range(sep_inst_enc.shape[2]):
        temp[:,:,i] = find_boundaries(sep_inst_enc[:,:,i], connectivity=1, mode='thick', background=0)
    
    # bc argmax will make the inst at 0 ch zeros so add a dummy channel
    dummy = np.zeros((temp.shape[0], temp.shape[1], 1))
    temp =  np.concatenate((dummy, temp), axis=-1)
    
    sep_inst_bdr = np.argmax(temp, axis=-1)
    sep_inst_bdr_rgb = gray2color(sep_inst_bdr, use_pallet='ade20k')
    if blend:
        inv = 1 - cv2.threshold(sep_inst_bdr.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, sep_inst_bdr_rgb)
    else:
        blend = sep_inst_bdr_rgb
        
    return blend

def get_sem(sem, img, blend=True, custom_pallet=None):
    '''
    Parameters
    ----------
    sem : a 2D array of shape [H, W] where containing unique value for each class.
    img : Original RGB image for overlaying the semantic seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each class have a unique RGB color. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    # if you get shape mismatch error try swaping the (w,h) argument of the line below.
    # i.e., from (x.shape[0], x.shape[1]) to (x.shape[1], x.shape[0]).
    img = cv2.resize(img, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR) 
    seg = gray2color(sem, use_pallet='pannuke', custom_pallet=custom_pallet)
    
    if blend:
        inv = 1 - cv2.threshold(sem.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, seg)
    else:
        blend = seg
        
    return blend


def get_sem_bdr(sem, img, blend=True, custom_pallet=None):
    '''
    Parameters
    ----------
    sem : a 2D array of shape [H, W] where containing unique value for each class.
    img : Original RGB image for overlaying the semantic seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each class have a unique RGB border. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    # if you get shape mismatch error try swaping the (w,h) argument of the line below.
    # i.e., from (x.shape[0], x.shape[1]) to (x.shape[1], x.shape[0]).
    img = cv2.resize(img, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR) 
    # 1-hot encode all classes 
    sem_enc = gray2encoded(sem, num_class=6)
    # as the in encoded output the 0th channel will be BG we don't need it so
    sem_enc = sem_enc[:,:,1:]
    # get boundaries of thest isolated instances
    temp = np.zeros(sem_enc.shape)
    
    for i in range(sem_enc.shape[2]):
        temp[:,:,i] = find_boundaries(sem_enc[:,:,i], connectivity=1, mode='thick', background=0)
    
    dummy = np.zeros((temp.shape[0], temp.shape[1], 1))
    temp =  np.concatenate((dummy, temp), axis=-1)
        
    sem_bdr = np.argmax(temp, axis=-1)
    sem_bdr_rgb = gray2color(sem_bdr, use_pallet='pannuke', custom_pallet=custom_pallet)
    if blend:
        inv = 1 - cv2.threshold(sem_bdr.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, sem_bdr_rgb)
    else:
        blend = sem_bdr_rgb
    return blend

def my_argmax(tensor):
    '''
    Fixes the zero channel problem i.e. the class predicted at 0th channel 
    wont go to 0 as it does with usual np.argmax
    Parameters
    ----------
    pred_tensor : 3D/4D array of shape [B, H, W, N] or [H, W, N]
    Returns
    -------
    argmaxed output of shape [B, H, W] or [H, W]]
    '''
    pred_tensor = np.copy(tensor)
    j = 0
    for i in range(pred_tensor.shape[-1]):
        j = i+1
        pred_tensor[:,:,:,i] = pred_tensor[:,:,:,i] * j
    
    pred_tensor = np.sum(pred_tensor, axis=-1)
    return pred_tensor    
