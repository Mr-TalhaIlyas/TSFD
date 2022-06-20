#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "-1";

import cv2, re, sys
import math, time, datetime
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tabulate import tabulate
from util import profile, keras_model_memory_usage, memory_usage, model_profile
from callbacks import WarmUpCosineDecayScheduler
from losses import Weighted_BCEnDice_loss, HED_loss, triplet_loss, FocalLoss, FocalTverskyLoss, TverskyLoss, SEG_Loss, INST_Loss
from metrics import dice_coef, mean_iou, binary_iou, Class_Wise_IOU, strict_iou
from plotting import plot_results, gray_2_rgb, tf_gray2rgb, prediction_writer, sigmoid_activation, softmax_activation
from data_gen import Tumor_Data_Generator, make_data_list, Tumor_IO, numericalSort
from morphology import bwmorph_thin, gray2encoded, seprate_instances
from pallet_n_classnames import pannuke_classes, pallet_pannuke
from models import num_of_classes, use_customdropout, efficent_pet_203_clf, efficent_pet_201, efficent_pet_203
'''
# Set the backend float after you call keras modules if called before keras modules will be called in their default config
keras.backend.floatx()
keras.backend.set_floatx('float16')
'''
import tensorflow as tf
from tensorflow import keras
tf_version = tf.__version__
if int(str(tf.__version__)[0]) == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  
    sess = tf.Session(config=config)
elif tf_version == '2.0.0':
    tf.compat.v1.disable_eager_execution()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # for gpus in physical_devices:
    #   tf.config.experimental.set_memory_growth(gpus, True)
      
elif int(str(tf.__version__)[0]) == 2:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # for gpus in physical_devices:
    #   tf.config.experimental.set_memory_growth(gpus, True)
    
if int(str(tf.__version__)[0]) == 2:
    import tensorflow_addons as tfa
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, Callback, TensorBoard
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import plot_model, multi_gpu_model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Dense

    print('\n Imorted libraries for tf.version = 2.x.x \n')
    
if int(str(tf.__version__)[0]) == 1:
    from keras.layers import Input
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, Callback
    from keras.optimizers import Adam, Nadam, SGD
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    from keras.utils import plot_model, multi_gpu_model
    from keras.regularizers import l2, l1
    from keras.layers.convolutional import Conv2D, SeparableConv2D, DepthwiseConv2D
    from keras.layers import Dense
    print('\n Imorted libraries for tf.version = 1.x.x \n')


# In[2]:


num_class = num_of_classes()
weight_name = 'Efficent_pet_203_clf'
log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M") + weight_name
Batch_size = 1
BiFPN_ch = 160
Epoch = 150
warmup_epoch = 3
im_width = 512
im_height = 512
d = 1 # GT mask downsampler
input_ch = 6
seed = 1
initial_lr = 0.16#0.005#0.0005
cycle = 100
mul_factor = 1.5
lr_decay = 0.9
drop_epoch = 5
power = 0.9
init_dropout = 0.2
Enc_Dropout_Type = 'SD'     #  'VD' -or- 'SD" -or- 'DB'
Dec_Dropout_Type = 'DB'     #  'VD' -or- 'SD" -or- 'DB'
'''
fliping dropout in enc and dec reduce accuracy
'''
dropout_schedule = np.array([init_dropout, 0.1, 0.15, 0.2, 0.3]) #@Epoch 0 10 20 30
dropout_after_epoch = np.array([0, 7, 14, 21, 30])
Blocksize = 5
use_mydropout = use_customdropout()
Weight_Decay = True 
w_decay_value = 0.0005
lr_schedule = 'SGDR_warmup'  # polynomial_decay  -or- step_decay -or- K_decay -or - SGDR_lr -or- SGDR_warmup
LOSS_function = '{Focal, Seg, Inst}'    # Weighted_BCEnDice_loss, focal_CE, HED_loss, triplet_loss
output_ch = num_class
class_names = ['Background', 'Inflammatory', 'Connective',
              'Dead ', 'Epithelial', 'Neoplastic ']
######################################################
#    Start Data Gen
######################################################
# train_dir = '/home/user01/data_ssd/Talha/pannuke/k_fold/k_fold_23/train/'
# val_dir = '/home/user01/data_ssd/Talha/pannuke/k_fold/k_fold_23/test/'
# test_dir = '/home/user01/data_ssd/Talha/pannuke/k_fold/k_fold_23/test/'

train_dir = '/home/user01/data_ssd/Talha/pannuke/train/'
val_dir = '/home/user01/data_ssd/Talha/pannuke/val/'
test_dir = '/home/user01/data_ssd/Talha/pannuke/test/'

train_imgs_paths, train_sem_masks_paths,  train_inst_masks_paths= make_data_list(train_dir)
val_imgs_paths, val_sem_masks_paths, val_inst_masks_paths = make_data_list(val_dir)
test_imgs_paths, test_sem_masks_paths, test_inst_masks_paths = make_data_list(test_dir)

tumor_types = np.unique(np.load('/home/user01/data_ssd/Talha/pannuke//types.npy'))# give the list of tumour types

train_data_gen = Tumor_Data_Generator(train_imgs_paths, train_sem_masks_paths, train_inst_masks_paths, tumor_types, Batch_size,
                                      im_width, im_height, data_augment=True, shuffle=True)
val_data_gen = Tumor_Data_Generator(val_imgs_paths, val_sem_masks_paths, val_inst_masks_paths, tumor_types, Batch_size,
                                      im_width, im_height, data_augment=False, shuffle=True)
test_data_gen = Tumor_Data_Generator(test_imgs_paths, test_sem_masks_paths, test_inst_masks_paths, tumor_types, Batch_size,
                                      im_width, im_height, data_augment=False, shuffle=True)

# to shuffle data at start of training
train_data_gen.on_epoch_end()
test_data_gen.on_epoch_end()
val_data_gen.on_epoch_end()


num_images = len(train_imgs_paths)
num_images_val = len(val_imgs_paths)
num_images_test = len(test_imgs_paths)

######################################################
# code sanity check
######################################################
z_image_hr, z_label_seg_clf = train_data_gen.__getitem__(28)
j = 0 
z_label_clf = z_label_seg_clf[0]
z_label_seg = z_label_seg_clf[1]
z_label_inst = z_label_seg_clf[2]
fig, ax = plt.subplots(1, 3)
ax[0].imshow(z_image_hr[j,:,:,:3])
ax[1].imshow(tf_gray2rgb(z_label_seg[j,:,:, 0], pallet_pannuke))
ax[1].set_title(tumor_types[np.argmax(z_label_clf[0,:])])
ax[2].imshow(z_label_inst[j,:,:,0], cmap='gray')
batchX = str((z_image_hr[0].shape))
batchY = str((z_label_inst.shape, z_label_seg.shape))


# In[3]:


input_img = Input((im_height, im_width, input_ch), name='ip')



model = efficent_pet_203_clf(input_img, output_ch, bifpn_ch = BiFPN_ch, dropout_rate=init_dropout, use_dropout=True)

model.compile(optimizer=SGD(momentum=0.9), loss={ FocalLoss,SEG_Loss, INST_Loss}, metrics={'accuracy', mean_iou, mean_iou}) 

# # print model summary on txt file
#model.summary()
orig_stdout = sys.stdout
f = open('/home/user01/data_ssd/Talha/pannuke/pan_final/efficent_pet_203.txt', 'w')
sys.stdout = f
print(model.summary())
sys.stdout = orig_stdout
f.close()
lr_scheduleW = lr_schedule 
trainable_param = model.count_params()
print('No. of parameters = ~',trainable_param/1000000,'Million')
table =  model_profile(model, Batch_size, initial_lr, w_decay_value, init_dropout, 
                  lr_scheduleW, Weight_Decay, use_mydropout, dropout_schedule, 
                  dropout_after_epoch, LOSS_function, batchX, batchY, 
                  Enc_Dropout_Type, Dec_Dropout_Type , Blocksize)




LR_schedule = WarmUpCosineDecayScheduler(learning_rate_base=initial_lr,
                                         total_steps=int(Epoch * num_images/Batch_size),
                                         warmup_learning_rate=0.0,
                                         warmup_steps=int(warmup_epoch * num_images/Batch_size))

log_tb = TensorBoard(
    log_dir='/home/user01/data_ssd/Talha/pannuke/pan_final/logs/{}/'.format(log_name))

print(table)

callbacks = [LR_schedule, log_tb, plot_losses,
            ModelCheckpoint('/home/user01/data_ssd/Talha/pannuke/pan_final/weights/{}.h5'.format(weight_name), verbose=1,
                            save_best_only=True, save_weights_only=True), # Save weights if val loss is improved
            CSVLogger('/home/user01/data_ssd/Talha/pannuke/pan_final/logs/{}.csv'.format(log_name), separator=',', append=True)
            ]  


# In[ ]:


results = model.fit(train_data_gen, steps_per_epoch= num_images/Batch_size, validation_data=val_data_gen, epochs=Epoch, 
                              initial_epoch = 0, validation_steps= num_images_val/Batch_size, callbacks=callbacks, verbose=1)


# In[ ]:


model.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/weights/{}.h5'.format(weight_name))


# In[4]:


model.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/weights/Efficent_pet_203_clf-end.h5')


# In[5]:


metrics = model.evaluate(test_data_gen)
result = np.concatenate((np.asarray(model.metrics_names).reshape(-1,1), np.round(np.asarray(metrics).reshape(-1,1),4)), 1)
print(tabulate(np.ndarray.tolist(result), headers = ["Metric", "Value"], tablefmt="github"))
x = np.asarray(metrics).reshape(-1,1)



