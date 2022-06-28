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

