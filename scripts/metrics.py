import tensorflow as tf
import numpy as np
from models import  num_of_classes
from tabulate import tabulate
if int(str(tf.__version__)[0]) == 1:
    from keras.layers.merge import concatenate
    from keras.layers import Activation
    import keras.backend as K
    from keras.backend import squeeze
if int(str(tf.__version__)[0]) == 2:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation, concatenate
#    import tensorflow_addons as tfa
    from tensorflow.keras.backend import squeeze

'''
  1.  In this on the fly lossses the ground truths are converted on they fly into the categorical type and only 
      hybrid and tri brid losses are doing that if you wann use only one loss then convert them first
  2. ALso read DATA set guidelines in MUST READ file
'''
#%%
num_class = num_of_classes()

def dice_coef(y_true, y_pred, smooth=2):
    
    if y_pred.shape[-1] <= 1:
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=[0])
    return dice
'''
In case of binary IoU both functions below work exactly the same 
    i.e. the number of op_channel == 1
'''
def mean_iou(y_true, y_pred, smooth=1):
    
    #y_true = y_true #* 255#(num_class + 1)
    if y_pred.shape[-1] <= 1:
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

def binary_iou(y_true, y_pred, smooth=1):
    
    if num_class < 2:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    else:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        
    y_true = y_true * (num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
    
    y_true = tf.cast(y_true, "int32")    
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    
    return iou
#%%
def Class_Wise_IOU(Pred, GT, NumClasses, ClassNames, display=False):
    '''
    Parameters
    ----------
    Pred: 2D array containing unique values for N classes [0, 1, 2, ...N]
    GT: 2D array containing unique values for N classes [0, 1, 2, ...N]
    NumClasses: int total number of classes including BG
    ClassNames : list of classes names
    Display: if want to print results
    Returns
    -------
    mean_IOU: mean over classes that are present in the GT (other classes are ignored)
    ClassIOU[:-1] : IOU of all classes in order
    ClassWeight[:-1] : no. of pixles in Union of each class present

    '''
    #Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate class intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            
    # b/c we will only take the mean over classes that are actually present in the GT
    present_classes = np.unique(GT) 
    mean_IOU = np.mean(ClassIOU[present_classes])
    # append it in final results
    ClassNames = np.append(ClassNames, 'Mean')
    ClassIOU = np.append(ClassIOU, mean_IOU)
    ClassWeight = np.append(ClassWeight, np.sum(ClassWeight))
    if display:
        result = np.concatenate((np.asarray(ClassNames).reshape(-1,1), 
                                 np.round(np.asarray(ClassIOU).reshape(-1,1),4),
                                 np.asarray(ClassWeight).reshape(-1,1)), 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Classes","IoU", "Class Weight(# Pixel)"], tablefmt="github"))
    
    return mean_IOU, ClassIOU[:-1], ClassWeight[:-1]

def Strict_IOU(Pred, GT, NumClasses, ClassNames):
    '''
    Parameters
    ----------
    Pred: 2D array containing unique values for N classes [0, 1, 2, ...N]
    GT: 2D array containing unique values for N classes [0, 1, 2, ...N]
    NumClasses: int total number of classes including BG
    ClassNames : list of classes names
    Display: if want to print results
    Returns
    -------
    mean_IOU: mean over classes that are present in the GT (other classes are ignored)
    ClassIOU[:-1] : IOU of all classes in order
    ClassWeight[:-1] : no. of pixles in Union of each class present

    '''
    #Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate class intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            
    # b/c we will only take the mean over classes that are actually present in the GT
    present_classes = np.unique(GT) 
    mean_IOU = np.mean(ClassIOU[present_classes])
    # append it in final results
    ClassNames = np.append(ClassNames, 'Mean')
    ClassIOU = np.append(ClassIOU, mean_IOU)
    ClassWeight = np.append(ClassWeight, np.sum(ClassWeight))
    
    return mean_IOU

NumClasses=6
ClassNames=['Background', 'Inflammatory', 'Connective',
            'Dead ', 'Epithelial', 'Neoplastic ']

def strict_iou(y_true, y_pred):
    
    '''
    only supported for btach size 1
    '''
    y_true = K.squeeze(y_true, 3)#[? H W 1] -> [? H W]
    y_true = K.squeeze(y_true, 0)#[H W] -> [H W]
    y_true = tf.cast(y_true, "int32")#[H W] -> [H W]
    
    
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)#[? H W Ch] -> [? H W Ch]
    y_pred = tf.cast(y_pred > 0.5, "int32")#[? H W Ch] -> [? H W Ch]
    y_pred = tf.math.argmax(y_pred, axis=-1)#[? H W CH] -> [? H W]
    y_pred = K.squeeze(y_pred, 0)#[? H W] -> [H W]
    
    x = tf.numpy_function(Strict_IOU, [y_pred, y_true, NumClasses, ClassNames], 
                          tf.float64, name=None)
    return x
