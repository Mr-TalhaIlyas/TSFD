import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add, multiply
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam, Nadam, SGD
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, BatchNormalization, Activation, ZeroPadding2D, Reshape, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply, add, PReLU, concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.backend import resize_images, int_shape
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Conv2DTranspose
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dropout, SpatialDropout2D



    
def SE_block(input_tensor, ratio = 8, activation = tf.nn.swish, Name=None):
    
    input_tensor = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='se_dw_{}'.format(Name))(input_tensor)
    input_tensor = BatchNormalization(name='se_bn_{}'.format(Name))(input_tensor)
    input_tensor = Activation(tf.nn.swish, name='se_act_{}'.format(Name))(input_tensor)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_tensor._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(name='gap_{}'.format(Name))(input_tensor)
    se = Reshape(se_shape, name='se_reshape_{}'.format(Name))(se)
    se = Dense(filters // ratio, activation=activation, kernel_initializer='he_normal', name='fc1_{}'.format(Name))(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', name='fc2_{}'.format(Name))(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([input_tensor, se], name='mul_{}'.format(Name))
    return x



def BiFPN(p2, p3, p4, p5, p6, p7, bifpn_layer, Dropout_Rate=0.3, Use_Dropout=False):

    
    p6_td = add([ SE_block(p6, Name='se_td_p6_{}'.format(bifpn_layer)),
                  SE_block(p7, Name='se_td_p7_{}'.format(bifpn_layer))], 
                  name='add_p6p7_{}'.format(bifpn_layer))
    
    p6_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    p6_td = BatchNormalization(name='bn_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    p6_td = Activation(tf.nn.swish, name='act_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    if Use_Dropout:
        p6_td = SpatialDropout2D(Dropout_Rate, name='drop_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
        
    
    p5_td = add([ SE_block(p5, Name='se_td_p5_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p6_td, mode='upsample'), Name='se_td_p6td_{}'.format(bifpn_layer))], 
                  name='add_p5p6td_{}'.format(bifpn_layer))
    
    p5_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    p5_td = BatchNormalization(name='bn_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    p5_td = Activation(tf.nn.swish, name='act_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    if Use_Dropout:
        p5_td = SpatialDropout2D(Dropout_Rate, name='drop_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
        
    # p4_td = conv[w*p4 + w*resize(p5_td)]
    p4_td = add([ SE_block(p4, Name='se_td_p4_{}'.format(bifpn_layer)),
                  SE_block(p5_td, Name='se_td_p5td_{}'.format(bifpn_layer))], 
                  name='add_p4p5td_{}'.format(bifpn_layer))
    
    p4_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    p4_td = BatchNormalization(name='bn_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    p4_td = Activation(tf.nn.swish, name='act_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    if Use_Dropout:
        p4_td = SpatialDropout2D(Dropout_Rate, name='drop_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    
    
    p3_td = add([ SE_block(p3, Name='se_td_p3_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p4_td, mode='upsample'), Name='se_td_p4td_{}'.format(bifpn_layer))], 
                  name='add_p3p4td_{}'.format(bifpn_layer))
    
    p3_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    p3_td = BatchNormalization(name='bn_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    p3_td = Activation(tf.nn.swish, name='act_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    if Use_Dropout:
        p3_td = SpatialDropout2D(Dropout_Rate, name='drop_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
        
    
    p2_out = add([ SE_block(p2, Name='se_td_p2_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p3_td, mode='upsample'), Name='se_td_p3td_{}'.format(bifpn_layer))], 
                  name='add_p2p3td_{}'.format(bifpn_layer))
    
    p2_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    p2_out = BatchNormalization(name='bn_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    p2_out = Activation(tf.nn.swish, name='act_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    if Use_Dropout:
        p2_out = SpatialDropout2D(Dropout_Rate, name='drop_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    '''
    Bottom-Up Path
    '''
   
    
    p3_out = add([ SE_block(p3, Name='se_bu_p3_{}'.format(bifpn_layer)),
                   SE_block(p3_td, Name='se_bu_p3td_{}'.format(bifpn_layer)),
                   SE_block(ReSize(p2_out, mode='downsample'), Name='se_bu_p2out_{}'.format(bifpn_layer))], 
                   name='add_p3p3tdp2out_{}'.format(bifpn_layer))
    
    p3_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p3out_bifpn_{}'.format(bifpn_layer))(p3_out)
    p3_out = BatchNormalization(name='bn_p3out_bifpn_{}'.format(bifpn_layer))(p3_out)
    p3_out = Activation(tf.nn.swish, name='act_p3out_bifpn_{}'.format(bifpn_layer))(p3_out)
    if Use_Dropout:
        p3_out = SpatialDropout2D(Dropout_Rate, name='drop_p3out_bifpn_{}'.format(bifpn_layer))(p3_out)
        
    p4_out = add([ SE_block(p4, Name='se_bu_p4_{}'.format(bifpn_layer)),
                   SE_block(p4_td, Name='se_bu_p4td_{}'.format(bifpn_layer)),
                   SE_block(ReSize(p3_out, mode='downsample'), Name='se_bu_p3out_{}'.format(bifpn_layer))], 
                   name='add_p4p4tdp3out_{}'.format(bifpn_layer))
    
    p4_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p4out_bifpn_{}'.format(bifpn_layer))(p4_out)
    p4_out = BatchNormalization(name='bn_p4out_bifpn_{}'.format(bifpn_layer))(p4_out)
    p4_out = Activation(tf.nn.swish, name='act_p4out_bifpn_{}'.format(bifpn_layer))(p4_out)
    if Use_Dropout:
        p4_out = SpatialDropout2D(Dropout_Rate, name='drop_p4out_bifpn_{}'.format(bifpn_layer))(p4_out)
        
    p5_out = add([ SE_block(p5, Name='se_bu_p5_{}'.format(bifpn_layer)),
                   SE_block(p5_td, Name='se_bu_p5td_{}'.format(bifpn_layer)),
                   SE_block(p4_out, Name='se_bu_p4out_{}'.format(bifpn_layer))], 
                   name='add_p5p5tdp4out_{}'.format(bifpn_layer))
    
    p5_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p5out_bifpn_{}'.format(bifpn_layer))(p5_out)
    p5_out = BatchNormalization(name='bn_p5out_bifpn_{}'.format(bifpn_layer))(p5_out)
    p5_out = Activation(tf.nn.swish, name='act_p5out_bifpn_{}'.format(bifpn_layer))(p5_out)
    if Use_Dropout:
        p5_out = SpatialDropout2D(Dropout_Rate, name='drop_p5out_bifpn_{}'.format(bifpn_layer))(p5_out)
        
    p6_out = add([ SE_block(p6, Name='se_bu_p6_{}'.format(bifpn_layer)),
                   SE_block(p6_td, Name='se_bu_p6td_{}'.format(bifpn_layer)),
                   SE_block(ReSize(p5_out, mode='downsample'), Name='se_bu_p5out_{}'.format(bifpn_layer))], 
                   name='add_p6p6tdp5out_{}'.format(bifpn_layer))
    
    p6_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p6out_bifpn_{}'.format(bifpn_layer))(p6_out)
    p6_out = BatchNormalization(name='bn_p6out_bifpn_{}'.format(bifpn_layer))(p6_out)
    p6_out = Activation(tf.nn.swish, name='act_p6out_bifpn_{}'.format(bifpn_layer))(p6_out)
    if Use_Dropout:
        p6_out = SpatialDropout2D(Dropout_Rate, name='drop_p6out_bifpn_{}'.format(bifpn_layer))(p6_out)
    
    p7_out = add([ SE_block(p7, Name='se_bu_p7_{}'.format(bifpn_layer)),
                   SE_block(p6_out, Name='se_bu_p6out_{}'.format(bifpn_layer))], 
                   name='add_p7p6out_{}'.format(bifpn_layer))
    
    p7_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p7out_bifpn_{}'.format(bifpn_layer))(p7_out)
    p7_out = BatchNormalization(name='bn_p7out_bifpn_{}'.format(bifpn_layer))(p7_out)
    p7_out = Activation(tf.nn.swish, name='act_p7out_bifpn_{}'.format(bifpn_layer))(p7_out)
    if Use_Dropout:
        p7_out = SpatialDropout2D(Dropout_Rate, name='drop_p7out_bifpn_{}'.format(bifpn_layer))(p7_out)
        
    return p2_out, p3_out, p4_out, p5_out, p6_out, p7_out


def Half_BiFPN(p2, p3, p4, p5, p6, p7, bifpn_layer, Dropout_Rate=0.1, Use_Dropout=False):

    '''
    top-Down Path
    '''
    
    p6_td = add([ SE_block(p6, Name='se_td_p6_{}'.format(bifpn_layer)),
                  SE_block(p7, Name='se_td_p7_{}'.format(bifpn_layer))], 
                  name='add_p6p7_{}'.format(bifpn_layer))
    
    p6_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    p6_td = BatchNormalization(name='bn_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    p6_td = Activation(tf.nn.swish, name='act_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
    if Use_Dropout:
        p6_td = SpatialDropout2D(Dropout_Rate, name='drop_p6td_bifpn_{}'.format(bifpn_layer))(p6_td)
        
    # p5_td = conv[w*p5 + w*resize(p6_td)]
    p5_td = add([ SE_block(p5, Name='se_td_p5_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p6_td, mode='upsample'), Name='se_td_p6td_{}'.format(bifpn_layer))], 
                  name='add_p5p6td_{}'.format(bifpn_layer))
    
    p5_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    p5_td = BatchNormalization(name='bn_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    p5_td = Activation(tf.nn.swish, name='act_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
    if Use_Dropout:
        p5_td = SpatialDropout2D(Dropout_Rate, name='drop_p5td_bifpn_{}'.format(bifpn_layer))(p5_td)
        
    
    p4_td = add([ SE_block(p4, Name='se_td_p4_{}'.format(bifpn_layer)),
                  SE_block(p5_td, Name='se_td_p5td_{}'.format(bifpn_layer))], 
                  name='add_p4p5td_{}'.format(bifpn_layer))
    
    p4_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    p4_td = BatchNormalization(name='bn_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    p4_td = Activation(tf.nn.swish, name='act_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    if Use_Dropout:
        p4_td = SpatialDropout2D(Dropout_Rate, name='drop_p4td_bifpn_{}'.format(bifpn_layer))(p4_td)
    
   
    p3_td = add([ SE_block(p3, Name='se_td_p3_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p4_td, mode='upsample'), Name='se_td_p4td_{}'.format(bifpn_layer))], 
                  name='add_p3p4td_{}'.format(bifpn_layer))
    
    p3_td = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    p3_td = BatchNormalization(name='bn_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    p3_td = Activation(tf.nn.swish, name='act_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
    if Use_Dropout:
        p3_td = SpatialDropout2D(Dropout_Rate, name='drop_p3td_bifpn_{}'.format(bifpn_layer))(p3_td)
        
    
    p2_out = add([ SE_block(p2, Name='se_td_p2_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p3_td, mode='upsample'), Name='se_td_p3td_{}'.format(bifpn_layer))], 
                  name='add_p2p3td_{}'.format(bifpn_layer))
    
    p2_out = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    p2_out = BatchNormalization(name='bn_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    p2_out = Activation(tf.nn.swish, name='act_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    if Use_Dropout:
        p2_out = SpatialDropout2D(Dropout_Rate, name='drop_p2out_bifpn_{}'.format(bifpn_layer))(p2_out)
    
    return p2_out


def OP_BiFPN(p2, p3, p4, p5, p6, p7, bifpn_layer, Dropout_Rate=0.3, Use_Dropout=False):

    '''
    Top-Down Path 
    '''
    p6_seg = add([ SE_block(p6, Name='se_seg_p6_{}'.format(bifpn_layer)),
                  SE_block(p7, Name='se_seg_p7_{}'.format(bifpn_layer))], 
                  name='add_p6p7_seg_{}'.format(bifpn_layer))
    
    p6_seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p6seg_bifpn_{}'.format(bifpn_layer))(p6_seg)
    p6_seg = BatchNormalization(name='bn_p6seg_bifpn_{}'.format(bifpn_layer))(p6_seg)
    p6_seg = Activation(tf.nn.swish, name='act_p6seg_bifpn_{}'.format(bifpn_layer))(p6_seg)
    if Use_Dropout:
        p6_seg = SpatialDropout2D(Dropout_Rate, name='drop_p6seg_bifpn_{}'.format(bifpn_layer))(p6_seg)
        
    p5_seg = add([ SE_block(p5, Name='se_seg_p5_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p6_seg, mode='upsample'), Name='se_seg_p6seg_{}'.format(bifpn_layer))], 
                  name='add_p5p6seg_{}'.format(bifpn_layer))
    
    p5_seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p5seg_bifpn_{}'.format(bifpn_layer))(p5_seg)
    p5_seg = BatchNormalization(name='bn_p5seg_bifpn_{}'.format(bifpn_layer))(p5_seg)
    p5_seg = Activation(tf.nn.swish, name='act_p5seg_bifpn_{}'.format(bifpn_layer))(p5_seg)
    if Use_Dropout:
        p5_seg = SpatialDropout2D(Dropout_Rate, name='drop_p5seg_bifpn_{}'.format(bifpn_layer))(p5_seg)
        
    p4_seg = add([ SE_block(p4, Name='se_seg_p4_{}'.format(bifpn_layer)),
                  SE_block(p5_seg, Name='se_seg_p5seg_{}'.format(bifpn_layer))], 
                  name='add_p4p5seg_{}'.format(bifpn_layer))
    
    p4_seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p4seg_bifpn_{}'.format(bifpn_layer))(p4_seg)
    p4_seg = BatchNormalization(name='bn_p4seg_bifpn_{}'.format(bifpn_layer))(p4_seg)
    p4_seg = Activation(tf.nn.swish, name='act_p4seg_bifpn_{}'.format(bifpn_layer))(p4_seg)
    if Use_Dropout:
        p4_seg = SpatialDropout2D(Dropout_Rate, name='drop_p4seg_bifpn_{}'.format(bifpn_layer))(p4_seg)
    
    p3_seg = add([ SE_block(p3, Name='se_seg_p3_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p4_seg, mode='upsample'), Name='se_seg_p4seg_{}'.format(bifpn_layer))], 
                  name='add_p3p4seg_{}'.format(bifpn_layer))
    
    p3_seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p3seg_bifpn_{}'.format(bifpn_layer))(p3_seg)
    p3_seg = BatchNormalization(name='bn_p3seg_bifpn_{}'.format(bifpn_layer))(p3_seg)
    p3_seg = Activation(tf.nn.swish, name='act_p3seg_bifpn_{}'.format(bifpn_layer))(p3_seg)
    if Use_Dropout:
        p3_seg = SpatialDropout2D(Dropout_Rate, name='drop_p3seg_bifpn_{}'.format(bifpn_layer))(p3_seg)
        
    p2_seg = add([ SE_block(p2, Name='se_seg_p2_{}'.format(bifpn_layer)),
                  SE_block(ReSize(p3_seg, mode='upsample'), Name='se_seg_p3seg_{}'.format(bifpn_layer))], 
                  name='add_p2p3seg_{}'.format(bifpn_layer))
    
    p2_seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p2segout_bifpn_{}'.format(bifpn_layer))(p2_seg)
    p2_seg = BatchNormalization(name='bn_p2segout_bifpn_{}'.format(bifpn_layer))(p2_seg)
    p2_seg = Activation(tf.nn.swish, name='act_p2segout_bifpn_{}'.format(bifpn_layer))(p2_seg)
    
    
    '''
    Top-Down Path 
    '''
    p6_inst = add([ SE_block(p7, Name='se_inst_p7_{}'.format(bifpn_layer)),
                    SE_block(p6_seg, Name='se_p6seg_inst_{}'.format(bifpn_layer))], 
                    name='add_p6p7p6seg_inst_{}'.format(bifpn_layer))
    
    p6_inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p6inst_bifpn_{}'.format(bifpn_layer))(p6_inst)
    p6_inst = BatchNormalization(name='bn_p6inst_bifpn_{}'.format(bifpn_layer))(p6_inst)
    p6_inst = Activation(tf.nn.swish, name='act_p6inst_bifpn_{}'.format(bifpn_layer))(p6_inst)
    if Use_Dropout:
        p6_inst = SpatialDropout2D(Dropout_Rate, name='drop_p6inst_bifpn_{}'.format(bifpn_layer))(p6_inst)
        
    p5_inst = add([ SE_block(ReSize(p6, mode='upsample'), Name='se_inst_p6_{}'.format(bifpn_layer)),
                    SE_block(ReSize(p6_inst, mode='upsample'), Name='se_inst_p6inst_{}'.format(bifpn_layer)),
                    SE_block(p5_seg, Name='se_p5seg_inst_{}'.format(bifpn_layer))], 
                    name='add_p5segp6p6inst_{}'.format(bifpn_layer))
    
    p5_inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p5inst_bifpn_{}'.format(bifpn_layer))(p5_inst)
    p5_inst = BatchNormalization(name='bn_p5inst_bifpn_{}'.format(bifpn_layer))(p5_inst)
    p5_inst = Activation(tf.nn.swish, name='act_p5inst_bifpn_{}'.format(bifpn_layer))(p5_inst)
    if Use_Dropout:
        p5_inst = SpatialDropout2D(Dropout_Rate, name='drop_p5inst_bifpn_{}'.format(bifpn_layer))(p5_inst)
        
    p4_inst = add([ SE_block(p5, Name='se_inst_p5_{}'.format(bifpn_layer)),
                    SE_block(p5_inst, Name='se_inst_p5inst_{}'.format(bifpn_layer)),
                    SE_block(p4_seg, Name='se_p4seg_inst_{}'.format(bifpn_layer))], 
                    name='add_p4segp5p5inst_{}'.format(bifpn_layer))
    
    p4_inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p4inst_bifpn_{}'.format(bifpn_layer))(p4_inst)
    p4_inst = BatchNormalization(name='bn_p4inst_bifpn_{}'.format(bifpn_layer))(p4_inst)
    p4_inst = Activation(tf.nn.swish, name='act_p4inst_bifpn_{}'.format(bifpn_layer))(p4_inst)
    if Use_Dropout:
        p4_inst = SpatialDropout2D(Dropout_Rate, name='drop_p4inst_bifpn_{}'.format(bifpn_layer))(p4_inst)
    
    p3_inst = add([ SE_block(ReSize(p4, mode='upsample'), Name='se_inst_p4_{}'.format(bifpn_layer)),
                    SE_block(ReSize(p4_inst, mode='upsample'), Name='se_inst_p4inst_{}'.format(bifpn_layer)),
                    SE_block(p3_seg, Name='se_p3seg_inst_{}'.format(bifpn_layer))], 
                    name='add_p3segp4p4inst_{}'.format(bifpn_layer))
    
    p3_inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p3inst_bifpn_{}'.format(bifpn_layer))(p3_inst)
    p3_inst = BatchNormalization(name='bn_p3inst_bifpn_{}'.format(bifpn_layer))(p3_inst)
    p3_inst = Activation(tf.nn.swish, name='act_p3inst_bifpn_{}'.format(bifpn_layer))(p3_inst)
    if Use_Dropout:
        p3_inst = SpatialDropout2D(Dropout_Rate, name='drop_p3inst_bifpn_{}'.format(bifpn_layer))(p3_inst)
        
    p2_inst = add([ SE_block(ReSize(p3, mode='upsample'), Name='se_inst_p3_{}'.format(bifpn_layer)),
                    SE_block(ReSize(p3_inst, mode='upsample'), Name='se_inst_p3inst_{}'.format(bifpn_layer)),
                    SE_block(p2_seg, Name='se_p2seg_inst_{}'.format(bifpn_layer))], 
                    name='add_p2segp3p3inst_{}'.format(bifpn_layer))
    
    p2_inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_p2instout_bifpn_{}'.format(bifpn_layer))(p2_inst)
    p2_inst = BatchNormalization(name='bn_p2instout_bifpn_{}'.format(bifpn_layer))(p2_inst)
    p2_inst = Activation(tf.nn.swish, name='act_p2instout_bifpn_{}'.format(bifpn_layer))(p2_inst)
    
        
    return p2_seg, p2_inst
