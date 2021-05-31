import tensorflow as tf
import numpy as np
from models import  num_of_classes
if int(str(tf.__version__)[0]) == 1:
    from keras.layers.merge import concatenate
    from keras.layers import Activation
    import keras.backend as K
if int(str(tf.__version__)[0]) == 2:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation, concatenate
    import tensorflow_addons as tfa

'''
In this on the fly lossses the ground truths are converted on they fly into the categorical type and only 
hybrid and tri brid losses are doing that if you wann use only one loss then convert them first
'''
'''
Clf losses
'''


def FocalLoss(y_true, y_pred):    
    alpha = 0.3#0.8
    gamma = 5
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss


'''
Seg losses
'''
#-------------------------------------------------------------Dice Loss Function-----------------------
num_class = num_of_classes()
#%%           Temporary test
alpha = 0.3
def SEG_Loss(y_true, y_pred):

    # weight fot tp hr v3 only
  loss = FocalTverskyLoss(y_true, y_pred, smooth=1e-6) + [0.4 * Weighted_BCEnDice_loss(y_true, y_pred)]

  return tf.math.reduce_mean(loss)
  
def INST_Loss(y_true, y_pred):
# weight fot tp hr v3 only
  loss = FocalTverskyLoss(y_true, y_pred, smooth=1e-6) + [0.4 * Combo_loss(y_true, y_pred)]

  return tf.math.reduce_mean(loss)


#%%

def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def Weighted_BCEnDice_loss(y_true, y_pred):
    
    if y_pred.shape[-1] <= 1:
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
       y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
       y_true = K.squeeze(y_true, 3)
       y_true = tf.cast(y_true, "int32")
       y_true = tf.one_hot(y_true, num_class, axis=-1)
       
   
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
#%%
def triplet_loss(y_true, y_pred):
    loss = Weighted_BCEnDice_loss(y_true, y_pred) + HED_loss(y_true, y_pred)
    return loss
#%%
def TverskyLoss(y_true, y_pred, smooth=1e-6):
    
        alpha = 0.3
        beta = 0.7
        
        if y_pred.shape[-1] <= 1:
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
           y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
           y_true = K.squeeze(y_true, 3)
           y_true = tf.cast(y_true, "int32")
           y_true = tf.one_hot(y_true, num_class, axis=-1)
           
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    

def FocalTverskyLoss(y_true, y_pred, smooth=1e-6):
        

        if y_pred.shape[-1] <= 1:
            alpha = 0.3
            beta = 0.7
            gamma = 5.
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
            alpha = 0.3
            beta = 0.7
            gamma = 3.
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)
        
        
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        Weighted_BCEnDice_loss
        return FocalTversky
        
#%%
'''
Dice similarity coefficient to deter model parameters from being
held at bad local minima and at the same time gradually learn better model 
parameters by penalizing for false positives/negatives
using a cross entropy term.
'''
def Combo_loss(y_true, y_pred, smooth=1e-6):
      
      e = K.epsilon()
      if y_pred.shape[-1] <= 1:
        ALPHA = 0.7#0.8    # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.4 # weighted contribution of modified CE loss compared to Dice loss
        y_pred = tf.keras.activations.sigmoid(y_pred)
      elif y_pred.shape[-1] >= 2:
        ALPHA = 0.7    # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.4 # weighted contribution of modified CE loss compared to Dice loss
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
      
      # cast to float32 datatype
      y_true = K.cast(y_true, 'float32')
      y_pred = K.cast(y_pred, 'float32')
      
      targets = K.flatten(y_true)
      inputs = K.flatten(y_pred)
      
      intersection = K.sum(targets * inputs)
      dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
      inputs = K.clip(inputs, e, 1.0 - e)
      out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
      weighted_ce = K.mean(out, axis=-1)
      combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
      
      return combo
#%%
  

def HED_loss(y_true, y_pred, smooth=1e-6):
    """
    From:

	  https://github.com/moabitcoin/holy-edge/blob/master/hed/losses.py

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    
    label = y_true 
    logits= y_pred 
    
    label = K.cast(label, 'float32')
    logits = K.cast(logits, 'float32')
    


    count_neg = tf.reduce_sum(1. - label)
    count_pos = tf.reduce_sum(label)

    # Equation [2]
    beta = (count_neg + smooth) / (count_neg + count_pos + smooth)

    # Equation [2] divide by 1 - beta
    pos_weight = (beta + smooth) / (1 - beta + smooth)
    if int(str(tf.__version__)[0]) == 1:
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
    if int(str(tf.__version__)[0]) == 2:
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=label, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)










    
    