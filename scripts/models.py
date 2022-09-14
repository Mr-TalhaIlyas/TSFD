import tensorflow as tf
if int(str(tf.__version__)[0]) == 2:
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, DepthwiseConv2D
    from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    import tensorflow.keras.backend as K
    from tensorflow.keras.applications import EfficientNetB4, EfficientNetB0
    from tensorflow.keras.layers.experimental.preprocessing import Resizing
    from tensorflow.keras.regularizers import L2
if int(str(tf.__version__)[0]) == 1:
    from keras.models import Model
    from keras.regularizers import l2, l1
    from keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, DepthwiseConv2D
    from keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, Dropout
    from keras.optimizers import Adam, Nadam, SGD
    import keras.backend as K

from bifpn import BiFPN, ReSize, Half_BiFPN, OP_BiFPN, Upsampling_block
import copy

        

'''
For binary segmentation set num_class=2
'''
def num_of_classes():
    num_class = 6
    return num_class

num_class = num_of_classes()
if num_class == 2:
    output_ch = 1
else:
    output_ch = num_class

if int(str(tf.__version__)[0]) == 1:
    from keras.layers import Dropout, SpatialDropout2D
if int(str(tf.__version__)[0]) == 2:
    from tensorflow.keras.layers import Dropout, SpatialDropout2D
 


#%%
def efficent_pet_203_clf(input_img, output_ch, bifpn_ch = 224, dropout_rate=0.3, use_dropout=False):
    
    '''
    Extract RGB features
    '''
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    #ef4.load_weights('../ef4.h5', by_name=True)
        
    p2 = ef4.layers[87].output 
    p3 = ef4.layers[146].output 
    p4 = ef4.layers[235].output 
    p5 = ef4.layers[323].output 
    p6 = ef4.layers[442].output 
    p7 = ef4.layers[473].output 
    
    '''
    Make Inputs for BiFPN
    '''
    
    
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    if use_dropout:
        p7 = SpatialDropout2D(dropout_rate, name='drop_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    if use_dropout:
        p6 = SpatialDropout2D(dropout_rate, name='drop_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    if use_dropout:
        p5 = SpatialDropout2D(dropout_rate, name='drop_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    if use_dropout:
        p4 = SpatialDropout2D(dropout_rate, name='drop_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    if use_dropout:
        p3 = SpatialDropout2D(dropout_rate, name='drop_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    if use_dropout:
        p2 = SpatialDropout2D(dropout_rate, name='drop_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, 
                                               bifpn_layer = 1, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, 
                                               bifpn_layer = 2, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, 
                                               bifpn_layer = 3, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    '''
    Top-Down FPNs for Seg and Inst branch
    '''
    p2_seg, p2_inst = OP_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='out')#, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    '''
    Seg Branch
    '''    
    p2_seg = Upsampling_block(p2_seg, output_ch)
    seg_output = DepthwiseConv2D((3, 3), padding='same', name='seg_out')(p2_seg)
    
    '''
    Inst Branch
    '''    
    p2_inst = Upsampling_block(p2_inst, 1)
    inst_output = DepthwiseConv2D((3, 3), padding='same', name='inst_out')(p2_inst)
    
    '''
    Clf Branch
    '''  
    gap = GlobalAveragePooling2D(name='GAP_clf')(p7)
    FC1 = Dropout(0.3)(gap)
    clf_output = Dense(19, activation='softmax', name='clf_out')(FC1) 
    model = Model(inputs=input_img, outputs=[clf_output, seg_output, inst_output])
    return model
